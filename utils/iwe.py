import torch


def purge_unfeasible(x, res):
    """
    Purge unfeasible event locations by setting their interpolation weights to zero.
    :param x: location of motion compensated events
    :param res: resolution of the image space
    :return masked indices
    :return mask for interpolation weights
    """

    mask = torch.ones((x.shape[0], x.shape[1], 1)).to(x.device)
    mask_y = (x[:, :, 0:1] < 0) + (x[:, :, 0:1] >= res[0])
    mask_x = (x[:, :, 1:2] < 0) + (x[:, :, 1:2] >= res[1])
    mask[mask_y + mask_x] = 0
    return x * mask, mask


def get_interpolation(events, flow, tref, res, flow_scaling, round_idx=False):
    """
    Warp the input events according to the provided optical flow map and compute the bilinar interpolation
    (or rounding) weights to distribute the events to the closes (integer) locations in the image space.
    :param events: [batch_size x N x 4] input events (ts, y, x, p)
    :param flow: [batch_size x N x 2] optical flows (y, x)
    :param tref: reference time toward which events are warped
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return interpolated event indices
    :return interpolation weights
    """

    # event propagation
    # flow scaling: max flow is largest dimension, combined with tanh this gets flow across entire sensor
    # (maybe not necessary for linear activation)
    warped_events = events[:, :, 1:3] + (tref - events[:, :, 0:1]) * flow * flow_scaling

    if round_idx:

        # no bilinear interpolation
        idx = torch.round(warped_events)
        weights = torch.ones(idx.shape).to(events.device)

    else:

        # get scattering indices
        top_y = torch.floor(warped_events[:, :, 0:1])
        bot_y = torch.floor(warped_events[:, :, 0:1] + 1)
        left_x = torch.floor(warped_events[:, :, 1:2])
        right_x = torch.floor(warped_events[:, :, 1:2] + 1)

        top_left = torch.cat([top_y, left_x], dim=2)
        top_right = torch.cat([top_y, right_x], dim=2)
        bottom_left = torch.cat([bot_y, left_x], dim=2)
        bottom_right = torch.cat([bot_y, right_x], dim=2)
        idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # get scattering interpolation weights
        warped_events = torch.cat([warped_events for i in range(4)], dim=1)
        zeros = torch.zeros(warped_events.shape).to(events.device)
        weights = torch.max(zeros, 1 - torch.abs(warped_events - idx))

    # purge unfeasible indices
    idx, mask = purge_unfeasible(idx, res)

    # make unfeasible weights zero
    weights = torch.prod(weights, dim=-1, keepdim=True) * mask  # bilinear interpolation

    # prepare indices
    idx[:, :, 0] *= res[1]  # torch.view is row-major
    idx = torch.sum(idx, dim=2, keepdim=True)

    return idx, weights


def interpolate(idx, weights, res, polarity_mask=None):
    """
    Create an image-like representation of the warped events.
    :param idx: [batch_size x N x 1] warped event locations
    :param weights: [batch_size x N x 1] interpolation weights for the warped events
    :param res: resolution of the image space
    :param polarity_mask: [batch_size x N x 1] polarity mask for the warped events (default = None)
    :return image of warped events
    """

    if polarity_mask is not None:
        weights = weights * polarity_mask
    iwe = torch.zeros((idx.shape[0], res[0] * res[1], 1)).to(idx.device)
    iwe = iwe.scatter_add_(1, idx.long(), weights)
    iwe = iwe.view((idx.shape[0], 1, res[0], res[1]))
    return iwe


def deblur_events(flow, event_list, res, flow_scaling=128, round_idx=True, polarity_mask=None):
    """
    Deblur the input events given an optical flow map.
    Event timestamp needs to be normalized between 0 and 1.
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param events: [batch_size x N x 4] input events (ts, y, x, p)
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :param polarity_mask: [batch_size x N x 1] polarity mask for the warped events (default = None)
    :return iwe: [batch_size x 1 x H x W] image of warped events
    """

    # flow vector per input event
    flow_idx = event_list[:, :, 1:3].clone()
    flow_idx[:, :, 0] *= res[1]  # torch.view is row-major
    flow_idx = torch.sum(flow_idx, dim=2)

    # get flow for every event in the list
    flow = flow.view(flow.shape[0], 2, -1)
    event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
    event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
    event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
    event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
    event_flow = torch.cat([event_flowy, event_flowx], dim=2)

    # interpolate forward
    fw_idx, fw_weights = get_interpolation(event_list, event_flow, 1, res, flow_scaling, round_idx=round_idx)
    if not round_idx and polarity_mask is not None:
        polarity_mask = torch.cat([polarity_mask for i in range(4)], dim=1)

    # image of (forward) warped events
    iwe = interpolate(fw_idx.long(), fw_weights, res, polarity_mask=polarity_mask)

    return iwe


def compute_pol_iwe(flow, event_list, res, pos_mask, neg_mask, flow_scaling=128, round_idx=True):
    """
    Create a per-polarity image of warped events given an optical flow map.
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
    :param res: resolution of the image space
    :param pos_mask: [batch_size x N x 1] polarity mask for positive events
    :param neg_mask: [batch_size x N x 1] polarity mask for negative events
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = True)
    :return iwe: [batch_size x 2 x H x W] image of warped events
    """

    iwe_pos = deblur_events(
        flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx, polarity_mask=pos_mask
    )
    iwe_neg = deblur_events(
        flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx, polarity_mask=neg_mask
    )
    iwe = torch.cat([iwe_pos, iwe_neg], dim=1)

    return iwe
