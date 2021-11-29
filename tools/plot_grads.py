import argparse

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def plot(grad_files):
    # loop over files
    for file in grad_files:
        # get data
        grads = pd.read_csv(file, index_col=0)
        # plot with rolling mean
        for name, grad in grads.iteritems():
            if "mean" in name:  # show only mean for now
                plt.plot(grad.rolling(100, min_periods=1).mean().to_numpy(), label=name)
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument("--files", nargs="+", default=["grads_w.csv"], help="filename(s) in artifact folder")
    args = parser.parse_args()

    # get filename(s)
    run = mlflow.get_run(args.runid)
    root = run.info.artifact_uri
    if root[:7] == "file://":
        root = root[7:]
    grad_files = []
    for file in args.files:
        grad_files.append(root + f"/{file}")

    plot(grad_files)
