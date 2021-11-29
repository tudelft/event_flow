"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch

from .base import BaseModel
from .model_util import copy_states, CropParameters
from .spiking_submodules import (
    ConvALIF,
    ConvALIFRecurrent,
    ConvLIF,
    ConvLIFRecurrent,
    ConvPLIF,
    ConvPLIFRecurrent,
    ConvXLIF,
    ConvXLIFRecurrent,
)
from .submodules import ConvGRU, ConvLayer, ConvLayer_, ConvLeaky, ConvLeakyRecurrent, ConvRecurrent
from .unet import (
    UNetRecurrent,
    MultiResUNet,
    MultiResUNetRecurrent,
    SpikingMultiResUNetRecurrent,
    LeakyMultiResUNetRecurrent,
)


class E2VID(BaseModel):
    """
    E2VID architecture (adapted for optical flow estimation), as described in the paper "High Speed and High
    Dynamic Range Video with an Event Camera", Rebecq et al., TPAMI 2021.
    """

    def __init__(self, unet_kwargs):
        super().__init__()

        norm = None
        use_upsample_conv = True
        if "norm" in unet_kwargs.keys():
            norm = unet_kwargs["norm"]
        if "use_upsample_conv" in unet_kwargs.keys():
            use_upsample_conv = unet_kwargs["use_upsample_conv"]

        E2VID_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 3,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "sum",
            "norm": norm,
            "use_upsample_conv": use_upsample_conv,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "recurrent_block_type": "convlstm",
            "final_activation": "tanh",
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = E2VID_kwargs["num_encoders"]

        unet_kwargs.update(E2VID_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("spiking_neuron", None)

        self.unetrecurrent = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def detach_states(self):
        detached_states = []
        for state in self.unetrecurrent.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.unetrecurrent.states = detached_states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_states

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        flow = self.unetrecurrent.forward(x)

        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        # crop output
        if self.crop is not None:
            flow = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
            flow = flow.contiguous()

        return {"flow": [flow], "activity": activity}


class FireNet(BaseModel):
    """
    FireNet architecture (adapted for optical flow estimation), as described in the paper "Fast Image
    Reconstruction with an Event Camera", Scheerlinck et al., WACV 2020.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvGRU
    residual = False
    num_recurrent_units = 7
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = None

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        ff_act, rec_act = unet_kwargs["activations"]
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        self.head = self.head_neuron(self.num_bins, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[0])

        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=rec_act, **self.kwargs[1]
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[2]
        )
        self.R1b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[3]
        )

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=rec_act, **self.kwargs[4]
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[5]
        )
        self.R2b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[6]
        )

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred
        )

        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # forward pass
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        x4, self._states[3] = self.R1b(x3, self._states[3], residual=x2 if self.residual else 0)

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6], residual=x5 if self.residual else 0)

        flow = self.pred(x7)

        # log activity
        if log:
            activity = {}
            name = [
                "0:input",
                "1:head",
                "2:G1",
                "3:R1a",
                "4:R1b",
                "5:G2",
                "6:R2a",
                "7:R2b",
                "8:pred",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, x6, x7, flow]):
                activity[n] = l.detach().ne(0).float().mean().item()
        else:
            activity = None

        return {"flow": [flow], "activity": activity}


class EVFlowNet(BaseModel):
    """
    EV-FlowNet architecture, as described in the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    def __init__(self, unet_kwargs):
        super().__init__()

        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": None,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]

        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("spiking_neuron", None)

        self.multires_unet = MultiResUNet(unet_kwargs)

    def detach_states(self):
        pass

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unet.forward(x)

        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        return {"flow": flow_list, "activity": activity}


class FireFlowNet(FireNet):
    """
    EV-FireFlowNet architecture, as described in the paper "Back to Event Basics: Self
    Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy",
    Paredes-Valles et al., CVPR 2021.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvLayer_
    residual = False
    w_scale_pred = 0.01


class RecEVFlowNet(BaseModel):
    """
    Recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = MultiResUNetRecurrent
    recurrent_block_type = "convgru"
    spiking_feedforward_block_type = None

    def __init__(self, unet_kwargs):
        super().__init__()

        norm = None
        use_upsample_conv = True
        if "norm" in unet_kwargs.keys():
            norm = unet_kwargs["norm"]
        if "use_upsample_conv" in unet_kwargs.keys():
            use_upsample_conv = unet_kwargs["use_upsample_conv"]

        RecEVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": norm,
            "use_upsample_conv": use_upsample_conv,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "recurrent_block_type": self.recurrent_block_type,
            "final_activation": "tanh",
            "spiking_feedforward_block_type": self.spiking_feedforward_block_type,
            "spiking_neuron": unet_kwargs["spiking_neuron"],
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = RecEVFlowNet_kwargs["num_encoders"]

        unet_kwargs.update(RecEVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("mask_output", None)

        self.multires_unetrec = self.unet_type(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.multires_unetrec.states)

    @states.setter
    def states(self, states):
        self.multires_unetrec.states = states

    def detach_states(self):
        detached_states = []
        for state in self.multires_unetrec.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.multires_unetrec.states = detached_states

    def reset_states(self):
        self.multires_unetrec.states = [None] * self.multires_unetrec.num_states

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unetrec.forward(x)

        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        return {"flow": flow_list, "activity": activity}


class SpikingRecEVFlowNet(RecEVFlowNet):
    """
    Spiking recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = SpikingMultiResUNetRecurrent
    recurrent_block_type = "lif"
    spiking_feedforward_block_type = "lif"


class PLIFRecEVFlowNet(RecEVFlowNet):
    """
    Spiking recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = SpikingMultiResUNetRecurrent
    recurrent_block_type = "plif"
    spiking_feedforward_block_type = "plif"


class ALIFRecEVFlowNet(RecEVFlowNet):
    """
    Spiking recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = SpikingMultiResUNetRecurrent
    recurrent_block_type = "alif"
    spiking_feedforward_block_type = "alif"


class XLIFRecEVFlowNet(RecEVFlowNet):
    """
    Spiking recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = SpikingMultiResUNetRecurrent
    recurrent_block_type = "xlif"
    spiking_feedforward_block_type = "xlif"


class RNNRecEVFlowNet(RecEVFlowNet):
    """
    Recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = MultiResUNetRecurrent
    recurrent_block_type = "convrnn"


class LeakyRecEVFlowNet(RecEVFlowNet):
    """
    Leaky recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = LeakyMultiResUNetRecurrent
    recurrent_block_type = "convleaky"


class RNNFireNet(FireNet):
    """
    Recurrent FireNet architecture of convolutional neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvRecurrent
    residual = False


class LeakyFireNet(FireNet):
    """
    Recurrent FireNet architecture of leaky/stateful convolutional neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvLeaky
    ff_neuron = ConvLeaky
    rec_neuron = ConvLeakyRecurrent
    residual = False


class LIFFireNet(FireNet):
    """
    Spiking FireNet architecture of LIF neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvLIF
    ff_neuron = ConvLIF
    rec_neuron = ConvLIFRecurrent
    residual = False
    w_scale_pred = 0.01


class PLIFFireNet(FireNet):
    """
    Spiking FireNet architecture of PLIF neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvPLIF
    ff_neuron = ConvPLIF
    rec_neuron = ConvPLIFRecurrent
    residual = False
    w_scale_pred = 0.01


class ALIFFireNet(FireNet):
    """
    Spiking FireNet architecture of ALIF neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvALIF
    ff_neuron = ConvALIF
    rec_neuron = ConvALIFRecurrent
    residual = False
    w_scale_pred = 0.01


class XLIFFireNet(FireNet):
    """
    Spiking FireNet architecture of XLIF neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvXLIF
    ff_neuron = ConvXLIF
    rec_neuron = ConvXLIFRecurrent
    residual = False
    w_scale_pred = 0.01


class LIFFireFlowNet(FireNet):
    """
    Spiking FireFlowNet architecture to investigate the power of implicit recurrency in SNNs.
    """

    head_neuron = ConvLIF
    ff_neuron = ConvLIF
    rec_neuron = ConvLIF
    residual = False


class LeakyFireFlowNet(FireNet):
    """
    FireFlowNet architecture with leaky internal state.
    """

    head_neuron = ConvLeaky
    ff_neuron = ConvLeaky
    rec_neuron = ConvLeaky
    residual = False
