import os

import mlflow
import pandas as pd
import torch


def load_model(prev_runid, model, device):
    try:
        run = mlflow.get_run(prev_runid)
    except:
        return model

    model_dir = run.info.artifact_uri + "/model/data/model.pth"
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device)
        model.load_state_dict(model_loaded.state_dict())
        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at" + prev_runid + "\n")

    return model


def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model):
    mlflow.pytorch.log_model(model, "model")


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")
