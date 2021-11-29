import os
import yaml

import torch
import mlflow


def log_config(path_results, runid, config):
    """
    Log configuration file to MlFlow run.
    """

    eval_id = 0
    for file in os.listdir(path_results):
        if file.endswith(".yml"):
            tmp = int(file.split(".")[0].split("_")[-1])
            eval_id = tmp + 1 if tmp + 1 > eval_id else eval_id
    yaml_filename = path_results + "eval_" + str(eval_id) + ".yml"
    with open(yaml_filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    mlflow.start_run(runid)
    mlflow.log_artifact(yaml_filename)
    mlflow.end_run()

    return eval_id


def log_results(runid, results, path, eval_id):
    """
    Log validation results as artifacts to MlFlow run.
    """

    yaml_filename = path + "metrics_" + str(eval_id) + ".yml"
    with open(yaml_filename, "w") as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    mlflow.start_run(runid)
    mlflow.log_artifact(yaml_filename)
    mlflow.end_run()
