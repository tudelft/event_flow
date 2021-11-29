import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import FWL, RSAT, AEE
from models.model import (
    FireNet,
    RNNFireNet,
    LeakyFireNet,
    FireFlowNet,
    LeakyFireFlowNet,
    E2VID,
    EVFlowNet,
    RecEVFlowNet,
    LeakyRecEVFlowNet,
    RNNRecEVFlowNet,
)
from models.model import (
    LIFFireNet,
    PLIFFireNet,
    ALIFFireNet,
    XLIFFireNet,
    LIFFireFlowNet,
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization, vis_activity


def test(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)

    # configs
    if config["loader"]["batch_size"] > 1:
        config["vis"]["enabled"] = False
        config["vis"]["store"] = False
        config["vis"]["bars"] = False  # progress bars not yet compatible batch_size > 1

    # asserts
    if "AEE" in config["metrics"]["name"]:
        assert (
            config["data"]["mode"] == "gtflow_dt1" or config["data"]["mode"] == "gtflow_dt4"
        ), "AEE computation not possible without ground truth mode"

    if "AEE" in config["metrics"]["name"]:
        assert config["data"]["window"] <= 1, "AEE computation not compatible with window > 1"
        assert np.isclose(
            (1.0 / config["data"]["window"]) % 1.0, 0.0
        ), "AEE computation not compatible with windows whose inverse is not a round number"

    if config["data"]["mode"] == "frames":
        if config["data"]["window"] <= 1.0:
            assert np.isclose(
                (1.0 / config["data"]["window"]) % 1.0, 0.0
            ), "Frames mode not compatible with < 1 windows whose inverse is not a round number"
        else:
            assert np.isclose(
                config["data"]["window"] % 1.0, 0.0
            ), "Frames mode not compatible with > 1 fractional windows"

    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)

        # store validation settings
        eval_id = log_config(path_results, args.runid, config)
    else:
        path_results = None
        eval_id = -1

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # model initialization and settings
    model = eval(config["model"]["name"])(config["model"]).to(device)
    model = load_model(args.runid, model, device)
    model.eval()

    # validation metric
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # data loader
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # inference loop
    idx_AEE = 0
    val_results = {}
    end_test = False
    activity_log = None
    with torch.no_grad():
        while True:
            for inputs in dataloader:

                if data.new_seq:
                    data.new_seq = False
                    activity_log = None
                    model.reset_states()

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # forward pass
                x = model(
                    inputs["event_voxel"].to(device), inputs["event_cnt"].to(device), log=config["vis"]["activity"]
                )

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask:
                    flow_vis *= inputs["event_mask"].to(device)

                # image of warped events
                iwe = compute_pol_iwe(
                    x["flow"][-1],
                    inputs["event_list"].to(device),
                    config["loader"]["resolution"],
                    inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                    inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                    flow_scaling=config["metrics"]["flow_scaling"],
                    round_idx=True,
                )

                iwe_window_vis = None
                events_window_vis = None
                masked_window_flow_vis = None
                if "metrics" in config.keys():

                    # event flow association
                    for metric in criteria:
                        metric.event_flow_association(x["flow"], inputs)

                    # validation
                    for i, metric in enumerate(config["metrics"]["name"]):
                        if criteria[i].num_events >= config["data"]["window_eval"]:

                            # overwrite intermedia flow estimates with the final ones
                            if config["loss"]["overwrite_intermediate"]:
                                criteria[i].overwrite_intermediate_flow(x["flow"])
                            if metric == "AEE" and inputs["dt_gt"] <= 0.0:
                                continue
                            if metric == "AEE":
                                idx_AEE += 1
                                if idx_AEE != np.round(1.0 / config["data"]["window"]):
                                    continue

                            # compute metric
                            val_metric = criteria[i]()
                            if metric == "AEE":
                                idx_AEE = 0

                            # accumulate results
                            for batch in range(config["loader"]["batch_size"]):
                                filename = data.files[data.batch_idx[batch] % len(data.files)].split("/")[-1]
                                if filename not in val_results.keys():
                                    val_results[filename] = {}
                                    for metric in config["metrics"]["name"]:
                                        val_results[filename][metric] = {}
                                        val_results[filename][metric]["metric"] = 0
                                        val_results[filename][metric]["it"] = 0
                                        if metric == "AEE":
                                            val_results[filename][metric]["percent"] = 0

                                val_results[filename][metric]["it"] += 1
                                if metric == "AEE":
                                    val_results[filename][metric]["metric"] += val_metric[0][batch].cpu().numpy()
                                    val_results[filename][metric]["percent"] += val_metric[1][batch].cpu().numpy()
                                else:
                                    val_results[filename][metric]["metric"] += val_metric[batch].cpu().numpy()

                            # visualize
                            if (
                                i == 0
                                and config["data"]["mode"] == "events"
                                and (config["vis"]["enabled"] or config["vis"]["store"])
                                and config["data"]["window"] < config["data"]["window_eval"]
                            ):
                                events_window_vis = criteria[i].compute_window_events()
                                iwe_window_vis = criteria[i].compute_window_iwe()
                                masked_window_flow_vis = criteria[i].compute_masked_window_flow()

                            # reset criteria
                            criteria[i].reset()

                # visualize
                if config["vis"]["bars"]:
                    for bar in data.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"]:
                    vis.update(inputs, flow_vis, iwe, events_window_vis, masked_window_flow_vis, iwe_window_vis)
                if config["vis"]["store"]:
                    sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                    vis.store(
                        inputs,
                        flow_vis,
                        iwe,
                        sequence,
                        events_window_vis,
                        masked_window_flow_vis,
                        iwe_window_vis,
                        ts=data.last_proc_timestamp,
                    )

                # visualize activity
                if config["vis"]["activity"]:
                    activity_log = vis_activity(x["activity"], activity_log)

            if end_test:
                break

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()

    # store validation config and results
    results = {}
    if not args.debug and "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric == "AEE":
                results[metric + "_percent"] = {}
            for key in val_results.keys():
                results[metric][key] = str(val_results[key][metric]["metric"] / val_results[key][metric]["it"])
                if metric == "AEE":
                    results[metric + "_percent"][key] = str(
                        val_results[key][metric]["percent"] / val_results[key][metric]["it"]
                    )
            log_results(args.runid, results, path_results, eval_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
