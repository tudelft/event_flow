from math import pi

import torch


def gaussian(x, mu, sigma):
    """
    Gaussian PDF with broadcasting.
    """
    return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * torch.sqrt(2 * torch.tensor(pi)))


class BaseSpike(torch.autograd.Function):
    """
    Baseline spiking function.
    """

    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x, width)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class SuperSpike(BaseSpike):
    """
    Spike function with SuperSpike surrogate gradient from
    "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.

    Design choices:
    - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
    - Width scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + width * x.abs()) ** 2
        return grad_input * sg, None


class MultiGaussSpike(BaseSpike):
    """
    Spike function with multi-Gaussian surrogate gradient from
    "Accurate and efficient time-domain classification...", Yin et al. 2021.

    Design choices:
    - Hyperparameters determined through grid search (Yin et al. 2021)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
        sg = (
            1.15 * gaussian(x, zero, width)
            - 0.15 * gaussian(x, width, 6 * width)
            - 0.15 * gaussian(x, -width, 6 * width)
        )
        return grad_input * sg, None


class TriangleSpike(BaseSpike):
    """
    Spike function with triangular surrogate gradient
    as in Bellec et al. 2020.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(1 - width * x.abs())
        return grad_input * sg, None


class ArctanSpike(BaseSpike):
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + width * x * x)
        return grad_input * sg, None


def superspike(x, thresh=torch.tensor(1.0), width=torch.tensor(10.0)):
    return SuperSpike.apply(x - thresh, width)


def mgspike(x, thresh=torch.tensor(1.0), width=torch.tensor(0.5)):
    return MultiGaussSpike.apply(x - thresh, width)


def trianglespike(x, thresh=torch.tensor(1.0), width=torch.tensor(1.0)):
    return TriangleSpike.apply(x - thresh, width)


def arctanspike(x, thresh=torch.tensor(1.0), width=torch.tensor(10.0)):
    return ArctanSpike.apply(x - thresh, width)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.linspace(-5, 5, 1001)

    superspike_ = 1 / (1 + 10 * x.abs()) ** 2

    zero = torch.tensor(0.0)
    sigma = torch.tensor(0.5)
    mgspike_ = (
        1.15 * gaussian(x, zero, sigma) - 0.15 * gaussian(x, sigma, 6 * sigma) - 0.15 * gaussian(x, -sigma, 6 * sigma)
    )

    fw = 1 / (1 + torch.exp(-10 * x))
    sigmoidspike_ = fw * (1 - fw)

    trianglespike_ = torch.nn.functional.relu(1 - x.abs())

    arctanspike_ = 1 / (1 + 10 * x * x)

    plt.plot(x.numpy(), superspike_.numpy(), label="superspike")
    plt.plot(x.numpy(), mgspike_.numpy(), label="mgspike")
    plt.plot(x.numpy(), sigmoidspike_.numpy(), label="sigmoidspike")
    plt.plot(x.numpy(), trianglespike_.numpy(), label="trianglespike")
    plt.plot(x.numpy(), arctanspike_.numpy(), label="arctanspike")
    plt.xlabel("v - thresh")
    plt.ylabel("grad")
    plt.grid()
    plt.legend()
    plt.show()
