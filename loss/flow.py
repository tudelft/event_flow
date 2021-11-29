import os
import sys

import numpy as np
import torch

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.iwe import purge_unfeasible, get_interpolation, interpolate


def spatial_variance(x):
    return torch.var(
        x.view(
            x.shape[0],
            x.shape[1],
            1,
            -1,
        ),
        dim=3,
        keepdim=True,
    )


class EventWarping(torch.nn.Module):
    """
    Contrast maximization loss, as described in Section 3.2 of the paper 'Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion', Zhu et al., CVPR'19.
    The contrast maximization loss is the minimization of the per-pixel and per-polarity image of averaged
    timestamps of the input events after they have been compensated for their motion using the estimated
    optical flow. This minimization is performed in a forward and in a backward fashion to prevent scaling
    issues during backpropagation.
    """

    def __init__(self, config, device, flow_scaling=None, loss_scaling=True):
        super(EventWarping, self).__init__()
        self.loss_scaling = loss_scaling
        self.res = config["loader"]["resolution"]
        self.flow_scaling = flow_scaling if flow_scaling is not None else max(config["loader"]["resolution"])
        self.weight = config["loss"]["flow_regul_weight"]
        self.smoothing_mask = False if "mask_output" not in config["model"].keys() else config["model"]["mask_output"]
        self.overwrite_intermediate = (
            False if "overwrite_intermediate" not in config["loss"].keys() else config["loss"]["overwrite_intermediate"]
        )
        self.device = device

        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_maps_x = None
        self._flow_maps_y = None
        self._pol_mask_list = None
        self._event_mask = None

    def event_flow_association(self, flow_list, event_list, pol_mask, event_mask):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param event_mask: [batch_size x 1 x H x W] event mask
        """

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        if self._flow_list is None:
            self._flow_list = []

        # get flow for every event in the list
        for i, flow in enumerate(flow_list):
            flow = flow.view(flow.shape[0], 2, -1)
            event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
            event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
            event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
            event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
            event_flow = torch.cat([event_flowy, event_flowx], dim=2)

            if i == len(self._flow_list):
                self._flow_list.append(event_flow)
            else:
                self._flow_list[i] = torch.cat([self._flow_list[i], event_flow], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list[:, :, 0:1] += self._passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        # update internal polarity mask list
        if self._pol_mask_list is None:
            self._pol_mask_list = pol_mask
        else:
            self._pol_mask_list = torch.cat([self._pol_mask_list, pol_mask], dim=1)

        # update internal smoothing mask
        if self._event_mask is None:
            self._event_mask = event_mask
        else:
            self._event_mask = torch.cat([self._event_mask, event_mask], dim=1)

        # update flow maps
        if self._flow_maps_x is None:
            self._flow_maps_x = []
            self._flow_maps_y = []

        for i, flow in enumerate(flow_list):
            if i == len(self._flow_maps_x):
                self._flow_maps_x.append(flow[:, 0:1, :, :])
                self._flow_maps_y.append(flow[:, 1:2, :, :])
            else:
                self._flow_maps_x[i] = torch.cat([self._flow_maps_x[i], flow[:, 0:1, :, :]], dim=1)
                self._flow_maps_y[i] = torch.cat([self._flow_maps_y[i], flow[:, 1:2, :, :]], dim=1)

        # update timestamp index
        self._passes += 1

    def overwrite_intermediate_flow(self, flow_list):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        """

        # flow vector per input event
        flow_idx = self._event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        self._flow_list = []
        self._flow_maps_x = []
        self._flow_maps_y = []

        # get flow for every event in the list
        for flow in flow_list:
            self._flow_maps_x.append(flow[:, 0:1, :, :])
            self._flow_maps_y.append(flow[:, 1:2, :, :])

            flow = flow.view(flow.shape[0], 2, -1)
            event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
            event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
            event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
            event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
            event_flow = torch.cat([event_flowy, event_flowx], dim=2)
            self._flow_list.append(event_flow)

        # update mask
        self._event_mask = torch.sum(self._event_mask, dim=1, keepdim=True)
        self._event_mask[self._event_mask > 1] = 1

    def reset(self):
        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_maps_x = None
        self._flow_maps_y = None
        self._pol_mask_list = None
        self._event_mask = None

    @property
    def num_events(self):
        if self._event_list is None:
            return 0
        else:
            return self._event_list.shape[1]

    @property
    def event_mask(self):
        if self.overwrite_intermediate:
            return self._event_mask  # mask of the training window
        else:
            return self._event_mask[:, -1:, :, :]  # mask of the last forward pass
        return self._event_mask

    def forward(self):
        max_ts = self._passes

        # split input
        pol_mask = torch.cat([self._pol_mask_list for i in range(4)], dim=1)
        ts_list = torch.cat([self._event_list[:, :, 0:1] for i in range(4)], dim=1)

        # smoothing mask
        if self.smoothing_mask:
            event_mask_dx = self._event_mask[:, :, :, :-1] * self._event_mask[:, :, :, 1:]
            event_mask_dy = self._event_mask[:, :, :-1, :] * self._event_mask[:, :, 1:, :]
            event_mask_dxdy_dr = self._event_mask[:, :, :-1, :-1] * self._event_mask[:, :, 1:, 1:]
            event_mask_dxdy_ur = self._event_mask[:, :, 1:, :-1] * self._event_mask[:, :, :-1, 1:]
            if not self.overwrite_intermediate:
                event_mask_dt = self._event_mask[:, :-1, :, :] * self._event_mask[:, 1:, :, :]

        loss = 0
        for i in range(len(self._flow_list)):

            # interpolate forward
            tref = max_ts
            fw_idx, fw_weights = get_interpolation(
                self._event_list, self._flow_list[i], tref, self.res, self.flow_scaling
            )

            # per-polarity image of (forward) warped events
            fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
            fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])

            # image of (forward) warped averaged timestamps
            fw_iwe_pos_ts = interpolate(
                fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 0:1]
            )
            fw_iwe_neg_ts = interpolate(
                fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 1:2]
            )
            fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
            fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
            fw_iwe_pos_ts = fw_iwe_pos_ts / max_ts
            fw_iwe_neg_ts = fw_iwe_neg_ts / max_ts

            # scale loss with number of pixels with at least one event in the image of warped events
            fw_iwe_pos_ts = fw_iwe_pos_ts.view(fw_iwe_pos_ts.shape[0], -1)
            fw_iwe_neg_ts = fw_iwe_neg_ts.view(fw_iwe_neg_ts.shape[0], -1)
            fw_loss = torch.sum(fw_iwe_pos_ts ** 2, dim=1) + torch.sum(fw_iwe_neg_ts ** 2, dim=1)
            if self.loss_scaling:
                fw_nonzero_px = fw_iwe_pos + fw_iwe_neg
                fw_nonzero_px[fw_nonzero_px > 0] = 1
                fw_nonzero_px = fw_nonzero_px.view(fw_nonzero_px.shape[0], -1)
                fw_loss /= torch.sum(fw_nonzero_px, dim=1)
            fw_loss = torch.sum(fw_loss)

            # interpolate backward
            tref = 0
            bw_idx, bw_weights = get_interpolation(
                self._event_list, self._flow_list[i], tref, self.res, self.flow_scaling
            )

            # per-polarity image of (backward) warped events
            bw_iwe_pos = interpolate(bw_idx.long(), bw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
            bw_iwe_neg = interpolate(bw_idx.long(), bw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])

            # image of (backward) warped averaged timestamps
            bw_iwe_pos_ts = interpolate(
                bw_idx.long(), bw_weights * (max_ts - ts_list), self.res, polarity_mask=pol_mask[:, :, 0:1]
            )
            bw_iwe_neg_ts = interpolate(
                bw_idx.long(), bw_weights * (max_ts - ts_list), self.res, polarity_mask=pol_mask[:, :, 1:2]
            )
            bw_iwe_pos_ts /= bw_iwe_pos + 1e-9
            bw_iwe_neg_ts /= bw_iwe_neg + 1e-9
            bw_iwe_pos_ts = bw_iwe_pos_ts / max_ts
            bw_iwe_neg_ts = bw_iwe_neg_ts / max_ts

            # scale loss with number of pixels with at least one event in the image of warped events
            bw_iwe_pos_ts = bw_iwe_pos_ts.view(bw_iwe_pos_ts.shape[0], -1)
            bw_iwe_neg_ts = bw_iwe_neg_ts.view(bw_iwe_neg_ts.shape[0], -1)
            bw_loss = torch.sum(bw_iwe_pos_ts ** 2, dim=1) + torch.sum(bw_iwe_neg_ts ** 2, dim=1)
            if self.loss_scaling:
                bw_nonzero_px = bw_iwe_pos + bw_iwe_neg
                bw_nonzero_px[bw_nonzero_px > 0] = 1
                bw_nonzero_px = bw_nonzero_px.view(bw_nonzero_px.shape[0], -1)
                bw_loss /= torch.sum(bw_nonzero_px, dim=1)
            bw_loss = torch.sum(bw_loss)

            # flow smoothing
            flow_x_dx = self._flow_maps_x[i][:, :, :, :-1] - self._flow_maps_x[i][:, :, :, 1:]
            flow_y_dx = self._flow_maps_y[i][:, :, :, :-1] - self._flow_maps_y[i][:, :, :, 1:]
            flow_x_dy = self._flow_maps_x[i][:, :, :-1, :] - self._flow_maps_x[i][:, :, 1:, :]
            flow_y_dy = self._flow_maps_y[i][:, :, :-1, :] - self._flow_maps_y[i][:, :, 1:, :]
            flow_x_dxdy_dr = self._flow_maps_x[i][:, :, :-1, :-1] - self._flow_maps_x[i][:, :, 1:, 1:]
            flow_y_dxdy_dr = self._flow_maps_y[i][:, :, :-1, :-1] - self._flow_maps_y[i][:, :, 1:, 1:]
            flow_x_dxdy_ur = self._flow_maps_x[i][:, :, 1:, :-1] - self._flow_maps_x[i][:, :, :-1, 1:]
            flow_y_dxdy_ur = self._flow_maps_y[i][:, :, 1:, :-1] - self._flow_maps_y[i][:, :, :-1, 1:]
            flow_x_dt = self._flow_maps_x[i][:, :-1, :, :] - self._flow_maps_x[i][:, 1:, :, :]
            flow_y_dt = self._flow_maps_y[i][:, :-1, :, :] - self._flow_maps_y[i][:, 1:, :, :]

            flow_dx = torch.sqrt((flow_x_dx + flow_y_dx) ** 2 + 1e-6)  # charbonnier
            flow_dy = torch.sqrt((flow_x_dy + flow_y_dy) ** 2 + 1e-6)  # charbonnier
            flow_dxdy_dr = torch.sqrt((flow_x_dxdy_dr + flow_y_dxdy_dr) ** 2 + 1e-6)  # charbonnier
            flow_dxdy_ur = torch.sqrt((flow_x_dxdy_ur + flow_y_dxdy_ur) ** 2 + 1e-6)  # charbonnier
            flow_dt = torch.sqrt((flow_x_dt + flow_y_dt) ** 2 + 1e-6)  # charbonnier

            # smoothing mask
            if self.smoothing_mask:
                flow_dx = event_mask_dx * flow_dx
                flow_dy = event_mask_dy * flow_dy
                flow_dxdy_dr = event_mask_dxdy_dr * flow_dxdy_dr
                flow_dxdy_ur = event_mask_dxdy_ur * flow_dxdy_ur
                if not self.overwrite_intermediate:
                    flow_dt = event_mask_dt * flow_dt

            components = 4
            smoothness_loss = flow_dx.sum() + flow_dy.sum() + flow_dxdy_dr.sum() + flow_dxdy_ur.sum()
            if not self.overwrite_intermediate:
                smoothness_loss += flow_dt.sum()
                components += 1
            smoothness_loss /= components
            smoothness_loss /= flow_dx.shape[1]

            loss += fw_loss + bw_loss + self.weight * smoothness_loss

        # average loss over all flow predictions
        loss /= len(self._flow_list)

        return loss


class BaseValidationLoss(torch.nn.Module):
    """
    Base class for validation metrics.
    """

    def __init__(self, config, device, flow_scaling=128):
        super(BaseValidationLoss, self).__init__()
        self.res = config["loader"]["resolution"]
        self.flow_scaling = flow_scaling  # should be specified by the user
        self.overwrite_intermediate = (
            False if "overwrite_intermediate" not in config["loss"].keys() else config["loss"]["overwrite_intermediate"]
        )
        self.device = device

        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._pol_mask_list = None
        self._event_mask = None

    @property
    def num_events(self):
        if self._event_list is None:
            return 0
        else:
            return self._event_list.shape[1]

    def event_flow_association(self, flow_list, inputs):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param inputs: dataloader dictionary
        """

        # move to device
        event_list = inputs["event_list"].to(self.device)
        pol_mask = inputs["event_list_pol_mask"].to(self.device)
        event_mask = inputs["event_mask"].to(self.device)
        gtflow = inputs["gtflow"].to(self.device) if "gtflow" in inputs.keys() else None

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow_list[-1]  # only highest resolution flow
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        if self._flow_list is None:
            self._flow_list = event_flow
        else:
            self._flow_list = torch.cat([self._flow_list, event_flow], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list = event_list.clone()  # to prevent issues with other metrics
            event_list[:, :, 0:1] += self._passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        # update internal polarity mask list
        if self._pol_mask_list is None:
            self._pol_mask_list = pol_mask
        else:
            self._pol_mask_list = torch.cat([self._pol_mask_list, pol_mask], dim=1)

        # update flow map
        if self._flow_map is None:
            self._flow_map = []
        self._flow_map.append(flow.view(flow.shape[0], 2, self.res[0], self.res[1]))

        # update ground-truth optical flow
        self._gtflow = gtflow

        # event mask
        if self._event_mask is None:
            self._event_mask = event_mask
        else:
            self._event_mask = torch.cat([self._event_mask, event_mask], dim=1)

        # update timestamps
        self._dt_input = inputs["dt_input"]
        self._dt_gt = inputs["dt_gt"]

        # update timestamp index
        self._passes += 1

    def overwrite_intermediate_flow(self, flow_list):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        """

        # flow vector per input event
        flow_idx = self._event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow_list[-1]  # only highest resolution flow
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        self._flow_list = event_flow
        self._flow_map = [flow.view(flow.shape[0], 2, self.res[0], self.res[1])]

        # update mask
        self._event_mask = torch.sum(self._event_mask, dim=1, keepdim=True)
        self._event_mask[self._event_mask > 1] = 1

    def reset(self):
        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._pol_mask_list = None
        self._event_mask = None

    def compute_window_events(self):
        idx = self._event_list[:, :, 1:3].clone()
        idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        idx = torch.sum(idx, dim=2, keepdim=True)
        weights = torch.ones(idx.shape).to(self.device)

        events_pos = interpolate(idx.long(), weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1])
        events_neg = interpolate(idx.long(), weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2])

        return torch.cat([events_pos, events_neg], dim=1)

    def compute_masked_window_flow(self):

        if self.overwrite_intermediate:
            return self._flow_map[-1] * self._event_mask
        else:
            avg_flow = self._flow_map[0] * self._event_mask[:, 0:1, :, :]
            for i in range(1, self._event_mask.shape[1]):
                avg_flow += self._flow_map[i] * self._event_mask[:, i : i + 1, :, :]
            avg_flow /= torch.sum(self._event_mask, dim=1, keepdim=True) + 1e-9
            return avg_flow

    def compute_window_iwe(self, round_idx=True):
        max_ts = self._passes
        if not round_idx:
            self._pol_mask_list = torch.cat([self._pol_mask_list for i in range(4)], dim=1)

        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=round_idx
        )
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2])

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)


class FWL(BaseValidationLoss):
    """
    Flow Warp Loss (FWL), as described in Section 4.1 of the paper 'Reducing the Sim-to-Real Gap for Event
    Cameras', Stoffregen and Scheerlinck et al., ECCV'20.
    The FWL metric is the ratio of the contrast of the image of warped events and that of the image of
    (non-warped) events; hence, the larger the value of this metric, the better the optical flow estimate.
    Contrast is measured through the spatial variance of the image of events.
    Note that this metric is sensitive to the number of input events.
    """

    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)

    def forward(self):
        max_ts = self._passes

        # image of (forward) warped events
        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        fw_IWE = interpolate(fw_idx.long(), fw_weights, self.res)

        # image of non-warped events
        zero_idx, zero_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        IE = interpolate(zero_idx.long(), zero_weights, self.res)

        # Forward Warping Loss (FWL)
        FWL = spatial_variance(fw_IWE) / spatial_variance(IE)
        FWL = FWL.view((fw_IWE.shape[0]))  # one metric per batch

        return FWL


class RSAT(BaseValidationLoss):
    """
    Similarly to the FWL metric, the Ratio of the Squared Averaged Timestamps (RSAT) metric is the ratio of the squared sum
    of the per-pixel and per-polarity average timestamp of the image of warped events and that of
    the image of (non-warped) events; hence, the lower the value of this metric, the better the optical flow estimate.
    Note that this metric is sensitive to the number of input events.
    """

    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)

    def forward(self):
        max_ts = self._passes

        # image of (forward) warped averaged timestamps
        ts_list = self._event_list[:, :, 0:1]
        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2])
        fw_iwe_pos_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        fw_iwe_neg_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
        fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
        fw_iwe_pos_ts = fw_iwe_pos_ts / max_ts
        fw_iwe_neg_ts = fw_iwe_neg_ts / max_ts

        # image of non-warped averaged timestamps
        zero_idx, zero_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        zero_iwe_pos = interpolate(
            zero_idx.long(), zero_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        zero_iwe_neg = interpolate(
            zero_idx.long(), zero_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        zero_iwe_pos_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        zero_iwe_neg_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        zero_iwe_pos_ts /= zero_iwe_pos + 1e-9
        zero_iwe_neg_ts /= zero_iwe_neg + 1e-9
        zero_iwe_pos_ts = zero_iwe_pos_ts / max_ts
        zero_iwe_neg_ts = zero_iwe_neg_ts / max_ts

        # (scaled) sum of the squares of the per-pixel and per-polarity average timestamps
        fw_iwe_pos_ts = fw_iwe_pos_ts.view(fw_iwe_pos_ts.shape[0], -1)
        fw_iwe_neg_ts = fw_iwe_neg_ts.view(fw_iwe_neg_ts.shape[0], -1)
        fw_iwe_pos_ts = torch.sum(fw_iwe_pos_ts ** 2, dim=1)
        fw_iwe_neg_ts = torch.sum(fw_iwe_neg_ts ** 2, dim=1)
        fw_ts_sum = fw_iwe_pos_ts + fw_iwe_neg_ts

        fw_nonzero_px = fw_iwe_pos + fw_iwe_neg
        fw_nonzero_px[fw_nonzero_px > 0] = 1
        fw_nonzero_px = fw_nonzero_px.view(fw_nonzero_px.shape[0], -1)
        fw_ts_sum /= torch.sum(fw_nonzero_px, dim=1)

        zero_iwe_pos_ts = zero_iwe_pos_ts.view(zero_iwe_pos_ts.shape[0], -1)
        zero_iwe_neg_ts = zero_iwe_neg_ts.view(zero_iwe_neg_ts.shape[0], -1)
        zero_iwe_pos_ts = torch.sum(zero_iwe_pos_ts ** 2, dim=1)
        zero_iwe_neg_ts = torch.sum(zero_iwe_neg_ts ** 2, dim=1)
        zero_ts_sum = zero_iwe_pos_ts + zero_iwe_neg_ts

        zero_nonzero_px = zero_iwe_pos + zero_iwe_neg
        zero_nonzero_px[zero_nonzero_px > 0] = 1
        zero_nonzero_px = zero_nonzero_px.view(zero_nonzero_px.shape[0], -1)
        zero_ts_sum /= torch.sum(zero_nonzero_px, dim=1)

        return fw_ts_sum / zero_ts_sum


class AEE(BaseValidationLoss):
    """
    Average endpoint error (which is just the Euclidean distance) loss.
    """

    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)

    @property
    def num_events(self):
        return float("inf")

    def forward(self):

        # convert flow
        flow = self._flow_map[-1] * self.flow_scaling
        flow *= self._dt_gt.to(self.device) / self._dt_input.to(self.device)
        flow_mag = flow.pow(2).sum(1).sqrt()

        # compute AEE
        error = (flow - self._gtflow).pow(2).sum(1).sqrt()

        # AEE not computed in pixels without events
        event_mask = self._event_mask[:, -1, :, :].bool()

        # AEE not computed in pixels without valid ground truth
        gtflow_mask_x = self._gtflow[:, 0, :, :] == 0.0
        gtflow_mask_y = self._gtflow[:, 1, :, :] == 0.0
        gtflow_mask = gtflow_mask_x * gtflow_mask_y
        gtflow_mask = ~gtflow_mask

        # mask AEE and flow
        mask = event_mask * gtflow_mask
        mask = mask.view(self._flow_map[-1].shape[0], -1)
        error = error.view(self._flow_map[-1].shape[0], -1)
        flow_mag = flow_mag.view(self._flow_map[-1].shape[0], -1)
        error = error * mask
        flow_mag = flow_mag * mask

        # compute AEE and percentage of outliers
        num_valid_px = torch.sum(mask, dim=1)
        AEE = torch.sum(error, dim=1) / (num_valid_px + 1e-9)

        outliers = (error > 3.0) * (error > 0.05 * flow_mag)  # AEE larger than 3px and 5% of the flow magnitude
        percent_AEE = outliers.sum() / (num_valid_px + 1e-9)

        return AEE, percent_AEE
