"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

import models.spiking_util as spiking


class ConvLayer(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
    ):
        super(ConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if w_scale is not None:
            nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
            nn.init.zeros_(self.conv2d.bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            else:
                self.activation = getattr(spiking, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConvLayer_(ConvLayer):
    """
    Clone of ConvLayer that acts like it has state, and allows residual.
    """

    def forward(self, x, prev_state, residual=0):
        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.tensor(0)  # not used

        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        out += residual
        if self.activation is not None:
            out = self.activation(out)

        return out, prev_state


class TransposedConvLayer(nn.Module):
    """
    Transposed convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation="relu",
        norm=None,
    ):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=bias,
        )

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            else:
                self.activation = getattr(spiking, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    """
    Upsampling layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
    ):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            else:
                self.activation = getattr(spiking, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        recurrent_block_type="convlstm",
        activation_ff="relu",
        activation_rec=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(RecurrentConvLayer, self).__init__()

        assert recurrent_block_type in ["convlstm", "convgru", "convrnn"]
        self.recurrent_block_type = recurrent_block_type
        if recurrent_block_type == "convlstm":
            RecurrentBlock = ConvLSTM
        elif recurrent_block_type == "convgru":
            RecurrentBlock = ConvGRU
        else:
            RecurrentBlock = ConvRecurrent

        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation_ff,
            norm,
            BN_momentum=BN_momentum,
        )
        self.recurrent_block = RecurrentBlock(
            input_size=out_channels, hidden_size=out_channels, kernel_size=3, activation=activation_rec
        )

    def forward(self, x, prev_state):
        x = self.conv(x)
        x, state = self.recurrent_block(x, prev_state)
        if isinstance(self.recurrent_block, ConvLSTM):
            state = (x, state)
        return x, state


class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="relu",
        downsample=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(ResidualBlock, self).__init__()
        bias = False if norm == "BN" else True
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            else:
                self.activation = getattr(spiking, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels, track_running_stats=True)
            self.bn2 = nn.InstanceNorm2d(out_channels, track_running_stats=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out1 = self.bn1(out1)

        if self.activation is not None:
            out1 = self.activation(out1)

        out2 = self.conv2(out1)
        if self.norm in ["BN", "IN"]:
            out2 = self.bn2(out2)

        if self.downsample:
            residual = self.downsample(x)

        out2 += residual
        if self.activation is not None:
            out2 = self.activation(out2)

        return out2, out1


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM module.
    Adapted from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2
        assert activation is None, "ConvLSTM activation cannot be set (just for compatibility)"

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state, new_state


class ConvRecurrent(nn.Module):
    """
    Convolutional recurrent cell (for direct comparison with spiking nets).
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()

        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.rec = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.out = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvRecurrent activation cannot be set (just for compatibility)"

    def forward(self, input_, prev_state):
        # generate empty prev_state, if None is provided
        if prev_state is None:
            batch, _, height, width = input_.shape
            state_shape = (batch, self.hidden_size, height, width)
            prev_state = torch.zeros(*state_shape, dtype=input_.dtype, device=input_.device)

        ff = self.ff(input_)
        rec = self.rec(prev_state)
        state = torch.tanh(ff + rec)
        out = self.out(state)
        out = torch.relu(out)

        return out, state


class ConvLeakyRecurrent(nn.Module):
    """
    Convolutional recurrent cell with leak (for direct comparison with spiking nets).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        activation=None,
        leak=(-4.0, 0.1),
        learn_leak=True,
        norm=None,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.rec = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.out = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        if learn_leak:
            self.leak = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        else:
            self.register_buffer("leak", torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        assert activation is None, "ConvLeakyRecurrent activation cannot be set (just for compatibility)"

    def forward(self, input_, prev_state):
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(*ff.shape, dtype=input_.dtype, device=input_.device)

        rec = self.rec(prev_state)  # call activation to get prev_out
        leak = torch.sigmoid(self.leak)
        state = prev_state * leak + (1 - leak) * (ff + rec)
        state = torch.tanh(state)
        out = self.out(state)
        out = torch.relu(out)

        return out, state


class ConvLeaky(nn.Module):
    """
    Convolutional stateful cell with leak (for direct comparison with spiking nets).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="relu",
        leak=(-4.0, 0.1),
        learn_leak=True,
        norm=None,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding)
        if learn_leak:
            self.leak = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        else:
            self.register_buffer("leak", torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            else:
                self.activation = getattr(spiking, activation)
        else:
            self.activation = None

    def forward(self, input_, prev_state, residual=0):
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(*ff.shape, dtype=input_.dtype, device=input_.device)

        leak = torch.sigmoid(self.leak)
        state = prev_state * leak + (1 - leak) * (ff + residual)

        if self.activation is not None:
            out = self.activation(state)
        else:
            out = state

        return out, state


class LeakyResidualBlock(nn.Module):
    """
    Leaky residual block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        feedforward_block_type="convleaky",
        activation="relu",
        **kwargs,
    ):
        super().__init__()

        assert feedforward_block_type in ["convleaky"]
        if feedforward_block_type == "convleaky":
            FeedforwardBlock = ConvLeaky

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


class LeakyUpsampleConvLayer(nn.Module):
    """
    Upsampling leaky layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        feedforward_block_type="convleaky",
        activation="relu",
        **kwargs,
    ):
        super().__init__()

        assert feedforward_block_type in ["convleaky"]
        if feedforward_block_type == "convleaky":
            FeedforwardBlock = ConvLeaky

        self.conv2d = FeedforwardBlock(
            in_channels, out_channels, kernel_size, stride=stride, activation=activation, **kwargs
        )

    def forward(self, x, prev_state):
        x_up = f.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x1, state = self.conv2d(x_up, prev_state)
        return x1, state


class LeakyTransposedConvLayer(nn.Module):
    """
    Transposed leaky convolutional layer to increase spatial resolution (x2) in a decoder.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        feedforward_block_type="convleaky",
        activation="relu",
        **kwargs,
    ):
        raise NotImplementedError


class LeakyRecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block,
    both leaky.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        recurrent_block_type="convleaky",
        activation_ff="relu",
        activation_rec=None,
        **kwargs,
    ):
        super().__init__()

        assert recurrent_block_type in ["convleaky"]
        if recurrent_block_type == "convleaky":
            FeedforwardBlock = ConvLeaky
            RecurrentBlock = ConvLeakyRecurrent

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
