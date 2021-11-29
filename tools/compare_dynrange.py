import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from models.spiking_submodules import ConvLIF, ConvALIF


def compare_dynrange(config_parser):

    # configs
    config = config_parser.config

    # initialize settings
    kwargs = config_parser.loader_kwargs

    # neuron
    in_channels = 2
    out_channels = 32
    kernel_size = 3

    # LIF
    var = "thresh"
    values = [0.1, 0.5, 0.8, 1.0]
    # var = "leak mean"
    # values = [-6.0, -4.0, -2.0, -1.0]

    # ALIF
    # var = "t0"
    # values = [0.01, 0.1, 0.2, 0.5]

    # inputs
    # steps: number of fw passes
    # window: scale/amount of input
    steps = 1
    window = [100, 1000, 2000, 4000, 5000, 10000, 20000, 40000, 50000]

    # compare different thresholds
    for v in values:

        in_mean_mean = []
        out_mean_mean = []
        out_std_mean = []

        # average over parameter inits
        for _ in range(5):

            neuron = ConvLIF(in_channels, out_channels, kernel_size, thresh=v)
            # neuron = ConvLIF(in_channels, out_channels, kernel_size, leak=(v, 0.1))
            # neuron = ConvALIF(in_channels, out_channels, kernel_size, t0=v)

            in_mean = []
            out_mean = []
            out_std = []

            with torch.no_grad():  # disable grads

                # compare different windows (input scale)
                for i, w in enumerate(window):

                    in_mean.append([])
                    out_mean.append([])
                    out_std.append([])
                    state = None

                    # configure data loader
                    config["data"]["window"] = w
                    data = H5Loader(config, 1)
                    dataloader = torch.utils.data.DataLoader(
                        data,
                        drop_last=True,
                        batch_size=1,
                        collate_fn=data.custom_collate,
                        worker_init_fn=config_parser.worker_init_fn,
                        **kwargs,
                    )

                    # get inputs
                    inputs = []
                    for inputs_ in dataloader:

                        inputs.append(inputs_["event_cnt"])
                        if len(inputs) == steps:
                            break

                    for inp in inputs:

                        z, state = neuron(inp, state)
                        in_mean[i].append(inp.mean((1, 2, 3)))  # mean over C, H, W
                        out_mean[i].append(z.mean((1, 2, 3)))  # mean over C, H, W
                        out_std[i].append(z.std((1, 2, 3)))  # std over C, H, W

                    in_mean[i] = torch.cat(in_mean[i], dim=0).mean(0).numpy()  # mean over time
                    out_mean[i] = torch.cat(out_mean[i], dim=0).mean(0).numpy()  # mean over time
                    out_std[i] = torch.cat(out_std[i], dim=0).mean(0).numpy()  # mean over time

            in_mean_mean.append(np.stack(in_mean))
            out_mean_mean.append(np.stack(out_mean))
            out_std_mean.append(np.stack(out_std))

        in_mean_mean = in_mean_mean[0]  # all the same
        out_mean_mean = np.stack(out_mean_mean).mean(0)  # mean over inits
        out_std_mean = np.stack(out_std_mean).mean(0)  # mean over inits

        plt.plot(window, out_mean_mean, label=f"{var}: {v}")
        plt.fill_between(window, out_mean_mean - out_std_mean, out_mean_mean + out_std_mean, alpha=0.2)

    plt.plot(window, in_mean_mean, "k", label="input")
    plt.grid()
    plt.legend()
    plt.xlabel("input window (# events)")
    plt.ylabel("activity (mean over C, H, W +- std)")
    plt.title(f"Activity after {steps} step(s) for {neuron._get_name()}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tools/compare_dynrange.yml", help="config file")
    args = parser.parse_args()

    # launch
    compare_dynrange(YAMLParser(args.config))
