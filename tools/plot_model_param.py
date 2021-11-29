import argparse
import sys

import matplotlib.pyplot as plt
import mlflow
import torch

sys.path.append(".")  # because file not in root
from models.model import FireFlowNet, EVFlowNet
from models.model import FireNet, E2VID, RecEVFlowNet, RNNFireNet, LeakyFireNet
from models.model import LIFFireNet, PLIFFireNet, ALIFFireNet, XLIFFireNet, LIFFireFlowNet
from utils.utils import load_model


def plot(model, to_plot):
    # loop over layers
    with torch.no_grad():
        for name, child in model.named_children():
            for param_name, param in child.named_parameters():
                if to_plot in param_name:
                    fig = plt.figure(param_name)
                    plt.hist(
                        param.view(-1).numpy(), density=True, alpha=0.5, edgecolor="k", label=f"{name}:{param_name}"
                    )
                    plt.legend()
                    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument("--params", nargs="+", default=["leak"], help="params to plot")
    args = parser.parse_args()

    # parse mlflow settings
    run = mlflow.get_run(args.runid).data.params
    config = {}
    for key in run.keys():
        if len(run[key]) > 0 and run[key][0] == "{":  # assume dictionary
            config[key] = eval(run[key])
        else:  # string
            config[key] = run[key]

    # add neuron config
    config["model"]["spiking_neuron"] = config["spiking_neuron"]

    # load model
    model = eval(config["model"]["name"])(config["model"])
    model = load_model(args.runid, model, "cpu")

    for p in args.params:
        plot(model, p)
