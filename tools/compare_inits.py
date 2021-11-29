import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


if __name__ == "__main__":

    in_channels = [2, 32, 64]
    out_channels = [2, 32, 64]
    kernel_size = 3

    for inc in in_channels:
        for outc in out_channels:
            default = nn.Conv2d(inc, outc, kernel_size).weight.detach()
            custom = torch.empty(outc, inc, kernel_size, kernel_size)

            scale = math.sqrt(1 / inc)  # zenke et al. 2021
            nn.init.uniform_(custom, -scale, scale)

            # plt.hist(default.sum((1, 2, 3)).numpy(), alpha=0.5, edgecolor="k", label=f"default {inc}-{outc}")
            plt.hist(custom.sum((1, 2, 3)).numpy(), alpha=0.5, edgecolor="k", label=f"custom {inc}-{outc}")

    plt.legend()
    plt.tight_layout()
    plt.show()
