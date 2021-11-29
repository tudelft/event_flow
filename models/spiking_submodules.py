import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.spiking_util as spiking


"""
Relevant literature:
- Zenke et al. 2018: "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"
- Bellec et al. 2020: "A solution to the learning dilemma for recurrent networks of spiking neurons"
- Fang et al. 2020: "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks"
- Ledinauskas et al. 2020: "Training Deep Spiking Neural Networks"
- Perez-Nieves et al. 2021: "Neural heterogeneity promotes robust learning"
- Yin et al. 2021: "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
- Zenke et al. 2021: "The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks"
- Fang et al. 2021: "Spike-based Residual Blocks"
- Paredes-Valles et al. 2020: "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception"
"""


class ConvLIF(nn.Module):
    """
    Convolutional spiking LIF cell.

    Design choices:
    - Arctan surrogate grad (Fang et al. 2021)
    - Hard reset (Ledinauskas et al. 2020)
    - Detach reset (Zenke et al. 2021)
    - Multiply previous voltage with leak; incoming current with (1 - leak) (Fang et al. 2020)
    - Make leak numerically stable with sigmoid (Fang et al. 2020)
    - Learnable threshold instead of bias
    - Per-channel leaks normally distributed (Yin et al. 2021)
    - Residual added to spikes (Fang et al. 2021)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak=(-4.0, 0.1),
        thresh=(0.8, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        if learn_leak:
            self.leak = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        else:
            self.register_buffer("leak", torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        if learn_thresh:
            self.thresh = nn.Parameter(torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])
        else:
            self.register_buffer("thresh", torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.norm = None
        elif norm == "group":
            groups = min(1, input_size // 4)  # at least instance norm
            self.norm = nn.GroupNorm(groups, input_size)
        else:
            self.norm = None

    def forward(self, input_, prev_state, residual=0):
        # input current
        if self.norm is not None:
            input_ = self.norm(input_)
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(2, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z = prev_state  # unbind op, removes dimension

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leak
        leak = torch.sigmoid(self.leak)

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak * (1 - z) + (1 - leak) * ff
        else:
            v_out = v * leak + (1 - leak) * ff - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out + residual, torch.stack([v_out, z_out])


class ConvPLIF(nn.Module):
    """
    Convolutional spiking LIF cell with adaptation based on pre-synaptic trace.
    Adapted from Paredes-Valles et al. 2020.

    Design choices: see ConvLIF.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_v=(-4.0, 0.1),
        leak_pt=(-4.0, 0.1),
        add_pt=(-2.0, 0.1),
        thresh=(0.8, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        self.pool = nn.AvgPool2d(kernel_size, stride, padding=padding)
        if learn_leak:
            self.leak_v = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.leak_pt = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
            self.add_pt = nn.Parameter(torch.randn(hidden_size, 1, 1) * add_pt[1] + add_pt[0])
        else:
            self.register_buffer("leak_v", torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.register_buffer("leak_pt", torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
            self.register_buffer("add_pt", torch.randn(hidden_size, 1, 1) * add_pt[1] + add_pt[0])
        if learn_thresh:
            self.thresh = nn.Parameter(torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])
        else:
            self.register_buffer("thresh", torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

    def forward(self, input_, prev_state, residual=0):
        # input current
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(3, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z, pt = prev_state  # unbind op, removes dimension

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leaks
        leak_v = torch.sigmoid(self.leak_v)
        leak_pt = torch.sigmoid(self.leak_pt)

        # get pt scaling
        add_pt = torch.sigmoid(self.add_pt)

        # pre-trace update: decay, add
        # mean of incoming channels, avg pooling over receptive field
        pt_out = pt * leak_pt + (1 - leak_pt) * self.pool(input_.abs().mean(1, keepdim=True))

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + (1 - leak_v) * (ff - add_pt * pt_out)
        else:
            v_out = v * leak_v + (1 - leak_v) * (ff - add_pt * pt_out) - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out + residual, torch.stack([v_out, z_out, pt_out])


class ConvALIF(nn.Module):
    """
    Convolutional spiking ALIF cell.

    Design choices:
    - Adaptive threshold (Bellec et al. 2020, Yin et al. 2021)
    - Parameters from Yin et al. 2021
    - Arctan surrogate grad (Fang et al. 2021)
    - Soft reset (Ledinauskas et al. 2020, Yin et al. 2021)
    - Detach reset (Zenke et al. 2021)
    - Multiply previous voltage with leak; incoming current with (1 - leak) (Fang et al. 2020)
    - Make leak numerically stable with sigmoid (Fang et al. 2020)
    - Per-channel leaks normally distributed (Yin et al. 2021)
    - Residual added to spikes (Fang et al. 2021)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_v=(-4.0, 0.1),
        leak_t=(-4.0, 0.1),
        t0=(0.01, 0.0),
        t1=(1.8, 0.0),
        learn_leak=True,
        learn_thresh=False,
        hard_reset=False,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        if learn_leak:
            self.leak_v = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.leak_t = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_t[1] + leak_t[0])
        else:
            self.register_buffer("leak_v", torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.register_buffer("leak_t", torch.randn(hidden_size, 1, 1) * leak_t[1] + leak_t[0])
        if learn_thresh:
            self.t0 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.t1 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])
        else:
            self.register_buffer("t0", torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.register_buffer("t1", torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

    def forward(self, input_, prev_state, residual=0):
        # input current
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(3, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z, t = prev_state  # unbind op, removes dimension

        # clamp thresh
        t0 = self.t0.clamp_min(0.01)
        t1 = self.t1.clamp_min(0)

        # get leaks
        leak_v = torch.sigmoid(self.leak_v)
        leak_t = torch.sigmoid(self.leak_t)

        # threshold update: decay, add
        t_out = t * leak_t + (1 - leak_t) * z
        # threshold: base + adaptive
        thresh = t0 + t1 * t_out

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + (1 - leak_v) * ff
        else:
            v_out = v * leak_v + (1 - leak_v) * ff - z * (t0 + t1 * t)

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out + residual, torch.stack([v_out, z_out, t_out])


class ConvXLIF(nn.Module):
    """
    Convolutional spiking LIF cell with threshold adaptation based on pre-synaptic trace.
    Crossing between PLIF and ALIF.

    Design choices: see ConvALIF.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_v=(-4.0, 0.1),
        leak_pt=(-4.0, 0.1),
        t0=(0.01, 0.0),
        t1=(1.8, 0.0),
        learn_leak=True,
        learn_thresh=False,
        hard_reset=False,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        self.pool = nn.AvgPool2d(kernel_size, stride, padding=padding)
        if learn_leak:
            self.leak_v = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.leak_pt = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
        else:
            self.register_buffer("leak_v", torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.register_buffer("leak_pt", torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
        if learn_thresh:
            self.t0 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.t1 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])
        else:
            self.register_buffer("t0", torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.register_buffer("t1", torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

    def forward(self, input_, prev_state, residual=0):
        # input current
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(3, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z, pt = prev_state  # unbind op, removes dimension

        # clamp thresh
        t0 = self.t0.clamp_min(0.01)
        t1 = self.t1.clamp_min(0)

        # get leaks
        leak_v = torch.sigmoid(self.leak_v)
        leak_pt = torch.sigmoid(self.leak_pt)

        # pre-trace update: decay, add
        # mean of incoming channels, avg pooling over receptive field
        pt_out = pt * leak_pt + (1 - leak_pt) * self.pool(input_.abs().mean(1, keepdim=True))
        # threshold: base + adaptive
        thresh = t0 + t1 * pt_out

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + (1 - leak_v) * ff
        else:
            v_out = v * leak_v + (1 - leak_v) * ff - z * (t0 + t1 * pt)

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out + residual, torch.stack([v_out, z_out, pt_out])


class ConvLIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking LIF cell.

    Design choices:
    - Arctan surrogate grad (Fang et al. 2021)
    - Hard reset (Ledinauskas et al. 2020)
    - Detach reset (Zenke et al. 2021)
    - Multiply previous voltage with leak; incoming current with (1 - leak) (Fang et al. 2020)
    - Make leak numerically stable with sigmoid (Fang et al. 2020)
    - Learnable threshold instead of bias
    - Per-channel leaks normally distributed (Yin et al. 2021)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        activation="arctanspike",
        act_width=10.0,
        leak=(-4.0, 0.1),
        thresh=(0.8, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding, bias=False)
        self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        if learn_leak:
            self.leak = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        else:
            self.register_buffer("leak", torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        if learn_thresh:
            self.thresh = nn.Parameter(torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])
        else:
            self.register_buffer("thresh", torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])

        # weight init
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.rec = nn.utils.weight_norm(self.rec)
            self.norm_ff = None
            self.norm_rec = None
        elif norm == "group":
            groups_ff = min(1, input_size // 4)  # at least instance norm
            groups_rec = min(1, hidden_size // 4)  # at least instance norm
            self.norm_ff = nn.GroupNorm(groups_ff, input_size)
            self.norm_rec = nn.GroupNorm(groups_rec, hidden_size)
        else:
            self.norm_ff = None
            self.norm_rec = None

    def forward(self, input_, prev_state):
        # input current
        if self.norm_ff is not None:
            input_ = self.norm_ff(input_)
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(2, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z = prev_state  # unbind op, removes dimension

        # recurrent current
        if self.norm_rec is not None:
            z = self.norm_rec(z)
        rec = self.rec(z)

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leak
        leak = torch.sigmoid(self.leak)

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak * (1 - z) + (1 - leak) * (ff + rec)
        else:
            v_out = v * leak + (1 - leak) * (ff + rec) - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out, torch.stack([v_out, z_out])


class ConvPLIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking LIF cell with adaptation based on pre-synaptic trace.
    Adapted from Paredes-Valles et al. 2020.

    Design choices: see ConvLIFRecurrent.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        activation="arctanspike",
        act_width=10.0,
        leak_v=(-4.0, 0.1),
        leak_pt=(-4.0, 0.1),
        add_pt=(-2.0, 0.1),
        thresh=(0.8, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding, bias=False)
        self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        if learn_leak:
            self.leak_v = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.leak_pt = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
            self.add_pt = nn.Parameter(torch.randn(hidden_size, 1, 1) * add_pt[1] + add_pt[0])
        else:
            self.register_buffer("leak_v", torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.register_buffer("leak_pt", torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
            self.register_buffer("add_pt", torch.randn(hidden_size, 1, 1) * add_pt[1] + add_pt[0])
        if learn_thresh:
            self.thresh = nn.Parameter(torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])
        else:
            self.register_buffer("thresh", torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])

        # weight init
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

    def forward(self, input_, prev_state, residual=0):
        # input current
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(3, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z, pt = prev_state  # unbind op, removes dimension

        # recurrent current
        rec = self.rec(z)

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leaks
        leak_v = torch.sigmoid(self.leak_v)
        leak_pt = torch.sigmoid(self.leak_pt)

        # get pt scaling
        add_pt = torch.sigmoid(self.add_pt)

        # pre-trace update: decay, add
        # mean of incoming channels, avg pooling over receptive field
        pt_out = pt * leak_pt + (1 - leak_pt) * self.pool(input_.abs().mean(1, keepdim=True))

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + (1 - leak_v) * (ff + rec - add_pt * pt_out)
        else:
            v_out = v * leak_v + (1 - leak_v) * (ff + rec - add_pt * pt_out) - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out + residual, torch.stack([v_out, z_out, pt_out])


class ConvALIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking ALIF cell.

    Design choices:
    - Adaptive threshold (Bellec et al. 2020, Yin et al. 2021)
    - Parameters from Yin et al. 2021
    - Arctan surrogate grad (Fang et al. 2021)
    - Soft reset (Ledinauskas et al. 2020, Yin et al. 2021)
    - Detach reset (Zenke et al. 2021)
    - Multiply previous voltage with leak; incoming current with (1 - leak) (Fang et al. 2020)
    - Make leak numerically stable with sigmoid (Fang et al. 2020)
    - Per-channel leaks normally distributed (Yin et al. 2021)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        activation="arctanspike",
        act_width=10.0,
        leak_v=(-4.0, 0.1),
        leak_t=(-4.0, 0.1),
        t0=(0.01, 0.0),
        t1=(1.8, 0.0),
        learn_leak=True,
        learn_thresh=False,
        hard_reset=False,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding, bias=False)
        self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        if learn_leak:
            self.leak_v = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.leak_t = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_t[1] + leak_t[0])
        else:
            self.register_buffer("leak_v", torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.register_buffer("leak_t", torch.randn(hidden_size, 1, 1) * leak_t[1] + leak_t[0])
        if learn_thresh:
            self.t0 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.t1 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])
        else:
            self.register_buffer("t0", torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.register_buffer("t1", torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])

        # weight init
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

    def forward(self, input_, prev_state):
        # input current
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(3, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z, t = prev_state  # unbind op, removes dimension

        # recurrent current
        rec = self.rec(z)

        # clamp thresh
        t0 = self.t0.clamp_min(0.01)
        t1 = self.t1.clamp_min(0)

        # get leaks
        leak_v = torch.sigmoid(self.leak_v)
        leak_t = torch.sigmoid(self.leak_t)

        # threshold update: decay, add
        t_out = t * leak_t + (1 - leak_t) * z
        # threshold: base + adaptive
        thresh = t0 + t1 * t_out

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + (1 - leak_v) * (ff + rec)
        else:
            v_out = v * leak_v + (1 - leak_v) * (ff + rec) - z * (t0 + t1 * t)

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out, torch.stack([v_out, z_out, t_out])


class ConvXLIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking LIF cell with threshold adaptation based on pre-synaptic trace.
    Crossing between PLIF and ALIF.

    Design choices: see ConvALIFRecurrent.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_v=(-4.0, 0.1),
        leak_pt=(-4.0, 0.1),
        t0=(0.01, 0.0),
        t1=(1.8, 0.0),
        learn_leak=True,
        learn_thresh=False,
        hard_reset=False,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        self.pool = nn.AvgPool2d(kernel_size, stride, padding=padding)
        if learn_leak:
            self.leak_v = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.leak_pt = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
        else:
            self.register_buffer("leak_v", torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
            self.register_buffer("leak_pt", torch.randn(hidden_size, 1, 1) * leak_pt[1] + leak_pt[0])
        if learn_thresh:
            self.t0 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.t1 = nn.Parameter(torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])
        else:
            self.register_buffer("t0", torch.randn(hidden_size, 1, 1) * t0[1] + t0[0])
            self.register_buffer("t1", torch.randn(hidden_size, 1, 1) * t1[1] + t1[0])

        # weight init
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

    def forward(self, input_, prev_state):
        # input current
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(3, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z, pt = prev_state  # unbind op, removes dimension

        # recurrent current
        rec = self.rec(z)

        # clamp thresh
        t0 = self.t0.clamp_min(0.01)
        t1 = self.t1.clamp_min(0)

        # get leaks
        leak_v = torch.sigmoid(self.leak_v)
        leak_pt = torch.sigmoid(self.leak_pt)

        # pre-trace update: decay, add
        # mean of incoming channels, avg pooling over receptive field
        pt_out = pt * leak_pt + (1 - leak_pt) * self.pool(input_.abs().mean(1, keepdim=True))
        # threshold: base + adaptive
        thresh = t0 + t1 * pt_out

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + (1 - leak_v) * (ff + rec)
        else:
            v_out = v * leak_v + (1 - leak_v) * (ff + rec) - z * (t0 + t1 * pt)

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out, torch.stack([v_out, z_out, pt_out])


class SpikingRecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block,
    both spiking. Default: no bias, arctanspike, no downsampling, no norm, LIF.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        recurrent_block_type="lif",
        activation_ff="arctanspike",
        activation_rec="arctanspike",
        **kwargs,
    ):
        super().__init__()

        assert recurrent_block_type in ["lif", "alif", "plif", "xlif"]
        if recurrent_block_type == "lif":
            FeedforwardBlock = ConvLIF
            RecurrentBlock = ConvLIFRecurrent
        elif recurrent_block_type == "alif":
            FeedforwardBlock = ConvALIF
            RecurrentBlock = ConvALIFRecurrent
        elif recurrent_block_type == "plif":
            FeedforwardBlock = ConvPLIF
            RecurrentBlock = ConvPLIFRecurrent
        else:
            FeedforwardBlock = ConvXLIF
            RecurrentBlock = ConvXLIFRecurrent
        kwargs.pop("spiking_feedforward_block_type", None)

        self.conv = FeedforwardBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation_ff,
            **kwargs,
        )
        self.recurrent_block = RecurrentBlock(
            out_channels, out_channels, kernel_size, activation=activation_rec, **kwargs
        )

    def forward(self, x, prev_state):
        if prev_state is None:
            prev_state = [None, None]
        ff, rec = prev_state  # unbind op, removes dimension
        x1, ff = self.conv(x, ff)
        x2, rec = self.recurrent_block(x1, rec)
        return x2, torch.stack([ff, rec])


class SpikingResidualBlock(nn.Module):
    """
    Spiking residual block as in "Spike-based Residual Blocks", Fang et al. 2021.
    Default: no bias, arctanspike, no downsampling, no norm, LIF.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        spiking_feedforward_block_type="lif",
        activation="arctanspike",
        **kwargs,
    ):
        super().__init__()

        assert spiking_feedforward_block_type in ["lif", "alif", "plif", "xlif"]
        if spiking_feedforward_block_type == "lif":
            FeedforwardBlock = ConvLIF
        elif spiking_feedforward_block_type == "alif":
            FeedforwardBlock = ConvALIF
        elif spiking_feedforward_block_type == "plif":
            FeedforwardBlock = ConvPLIF
        else:
            FeedforwardBlock = ConvXLIF

        self.conv1 = FeedforwardBlock(
            in_channels, out_channels, kernel_size=3, stride=stride, activation=activation, **kwargs
        )
        self.conv2 = FeedforwardBlock(
            out_channels, out_channels, kernel_size=3, stride=1, activation=activation, **kwargs
        )

    def forward(self, x, prev_state):
        if prev_state is None:
            prev_state = [None, None]
        conv1, conv2 = prev_state  # unbind op, removes dimension

        residual = x
        x1, conv1 = self.conv1(x, conv1)
        x2, conv2 = self.conv2(x1, conv2, residual=residual)  # add res inside
        return x2, torch.stack([conv1, conv2])


class SpikingUpsampleConvLayer(nn.Module):
    """
    Upsampling spiking layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    Default: no bias, arctanspike, no downsampling, no norm, LIF.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        spiking_feedforward_block_type="lif",
        activation="arctanspike",
        **kwargs,
    ):
        super().__init__()

        assert spiking_feedforward_block_type in ["lif", "alif", "plif", "xlif"]
        if spiking_feedforward_block_type == "lif":
            FeedforwardBlock = ConvLIF
        elif spiking_feedforward_block_type == "alif":
            FeedforwardBlock = ConvALIF
        elif spiking_feedforward_block_type == "plif":
            FeedforwardBlock = ConvPLIF
        else:
            FeedforwardBlock = ConvXLIF

        self.conv2d = FeedforwardBlock(
            in_channels, out_channels, kernel_size, stride=stride, activation=activation, **kwargs
        )

    def forward(self, x, prev_state):
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x1, state = self.conv2d(x_up, prev_state)
        return x1, state


class SpikingTransposedConvLayer(nn.Module):
    """
    Transposed spiking convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: no bias, arctanspike, no downsampling, no norm, LIF.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        spiking_feedforward_block_type="lif",
        activation="arctanspike",
        **kwargs,
    ):
        raise NotImplementedError
