import os
import sys
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from utils.iwe import compute_pol_iwe
from utils.visualization import Visualization


def demo(args, config_parser):

    # configs
    config = config_parser.config

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    vis = Visualization(config)

    # data loader
    data = H5Loader(config, 1)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=1,
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # loss function
    loss_function = EventWarping(config, device, flow_scaling=1)

    # heatmap elements
    x_scale = np.linspace(-args.x_maxdisp, args.x_maxdisp, num=args.heatmap_res)
    y_scale = np.linspace(-args.y_maxdisp, args.y_maxdisp, num=args.heatmap_res)
    flowmap = torch.ones((1, 2, config["loader"]["resolution"][0], config["loader"]["resolution"][1])).to(device)

    # loop
    end_test = False
    with torch.no_grad():
        while True:
            for inputs in dataloader:

                if data.new_seq:
                    data.new_seq = False

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # check for keyboard input
                key = cv2.waitKey(args.sleep_time)
                if key != -1 and key == ord("p"):

                    # compute heatmap
                    heatmap = np.zeros((args.heatmap_res, args.heatmap_res))
                    for i, u in enumerate(x_scale):
                        for j, v in enumerate(y_scale):
                            flow = flowmap.clone()
                            flow[:, 0, :, :] *= u
                            flow[:, 1, :, :] *= v

                            loss_function.event_flow_association(
                                [flow],
                                inputs["event_list"].to(device),
                                inputs["event_list_pol_mask"].to(device),
                                inputs["event_mask"].to(device),
                            )
                            loss = loss_function()
                            loss_function.reset()
                            heatmap[j, i] = loss.cpu().numpy()

                    # flow with lowest error
                    index_best = np.unravel_index(np.argmin(heatmap, axis=None), heatmap.shape)
                    flow = flowmap.clone()
                    flow[:, 0, :, :] *= x_scale[index_best[0]]
                    flow[:, 1, :, :] *= y_scale[index_best[1]]

                    # optimal image of warped events
                    iwe = compute_pol_iwe(
                        flow,
                        inputs["event_list"].to(device),
                        config["loader"]["resolution"],
                        inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                        inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                        flow_scaling=1,
                        round_idx=True,
                    )

                    # visualize
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                    cv2.namedWindow("Heatmap", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Heatmap", int(config["vis"]["px"]), int(config["vis"]["px"]))
                    cv2.imshow("Heatmap", heatmap)
                    vis.update(inputs, None, iwe)
                    cv2.waitKey(0)

                # visualize
                vis.update(inputs, None, None)

            if end_test:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tools/demo_iwe.yml", help="config file")
    parser.add_argument("--sleep_time", default=50, help="sleep time for keyboard input")
    parser.add_argument("--heatmap_res", default=50, help="heatmap resolution")
    parser.add_argument("--x_maxdisp", default=64, help="heatmap resolution")
    parser.add_argument("--y_maxdisp", default=64, help="heatmap resolution")
    args = parser.parse_args()

    # launch demo
    demo(args, YAMLParser(args.config))
