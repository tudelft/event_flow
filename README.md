# Self-Supervised Learning of Event-based Optical Flow with Spiking Neural Networks

Work accepted at NeurIPS'21 [[paper](https://proceedings.neurips.cc/paper/2021/hash/39d4b545fb02556829aab1db805021c3-Abstract.html), [video](https://www.youtube.com/watch?v=T7-9GGYnuZ4&ab_channel=MAVLabTUDelft)].

If you use this code in an academic context, please cite our work:

```bibtex
@article{hagenaarsparedesvalles2021ssl,
  title={Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks},
  author={Hagenaars, Jesse and Paredes-Vall\'es, Federico and de Croon, Guido},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

This code allows for the reproduction of the experiments leading to the results in Section 4.1.

<!-- &nbsp; -->
<img src=".readme/flow.gif" width="880" height="220" />
<!-- &nbsp; -->

#

## Usage

This project uses Python >= 3.7.3 and we strongly recommend the use of virtual environments. If you don't have an environment manager yet, we recommend `pyenv`. It can be installed via:

```
curl https://pyenv.run | bash
```

Make sure your `~/.bashrc` file contains the following:

```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

After that, restart your terminal and run:

```
pyenv update
```

To set up your environment with `pyenv` first install the required python distribution and make sure the installation is successful (i.e., no errors nor warnings):

```
pyenv install -v 3.7.3
```

Once this is done, set up the environment and install the required libraries:

```
pyenv virtualenv 3.7.3 event_flow
pyenv activate event_flow

pip install --upgrade pip==20.0.2

cd event_flow/
pip install -r requirements.txt
```

### Download datasets

In this work, we use multiple datasets:
- `event_flow/datasets/data/training`: [UZH-FPV Drone Racing Dataset](https://fpv.ifi.uzh.ch/) (Delmerico, ICRA'19)
- `event_flow/datasets/data/MVSEC`: [Multi View Stereo Event Camera Dataset](https://daniilidis-group.github.io/mvsec/) (Zhu, RA-L'18)
- `event_flow/datasets/data/ECD`: [Event-Camera Dataset](http://rpg.ifi.uzh.ch/davis_data.html) (Mueggler, IJRR'17)
- `event_flow/datasets/data/HQF`: [High Quality Frames](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720528.pdf) (Stoffregen and Scheerlinck, ECCV'20)

These datasets can be downloaded in the expected HDF5 data format from [here](https://surfdrive.surf.nl/files/index.php/s/4xXLV89pt2IphnB), and are expected at `event_flow/datasets/data/` (as shown above). 

Download size: 19.4 GB. Uncompressed size: 94 GB.

Details about the structure of these files can be found in `event_flow/datasets/tools/`. 

### Download models

The pretrained models can be downloaded from [here](https://surfdrive.surf.nl/files/index.php/s/EX55MmYrUB8ExVx), and are expected at `event_flow/mlruns/`. 

In this project we use [MLflow](https://www.mlflow.org/docs/latest/index.html#) to keep track of the experiments. To visualize the models that are available, alongside other useful details and evaluation metrics, run the following from the home directory of the project:

```
mlflow ui
```

and access [http://127.0.0.1:5000](http://127.0.0.1:5000) from your browser of choice.

## Inference

To estimate optical flow from event sequences from the MVSEC dataset and compute the average endpoint error and percentage of outliers, run:

```
python eval_flow.py <model_name> --config configs/eval_MVSEC.yml

# for example:
python eval_flow.py LIFFireNet --config configs/eval_MVSEC.yml
```

where `<model_name>` is the name of MLflow run to be evaluated. Note that, if a run does not have a name (this would be the case for your own trained models), you can evaluated it through its run ID (also visible through MLflow).

To estimate optical flow from event sequences from the ECD or HQF datasets, run:

```
python eval_flow.py <model_name> --config configs/eval_ECD.yml
python eval_flow.py <model_name> --config configs/eval_HQF.yml

# for example:
python eval_flow.py LIFFireNet --config configs/eval_ECD.yml
```

Note that the ECD and HQF datasets lack ground truth optical flow data. Therefore, we evaluate the quality of the estimated event-based optical flow via the self-supervised [FWL](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720528.pdf) (Stoffregen and Scheerlinck, ECCV'20) and RSAT (ours, Appendix C) metrics.

Results from these evaluations are stored as MLflow artifacts. 

In `configs/`, you can find the configuration files associated to these scripts and vary the inference settings (e.g., number of input events, activate/deactivate visualization).

## Training

Run:

```
python train_flow.py --config configs/train_ANN.yml
python train_flow.py --config configs/train_SNN.yml
```

to train an traditional artificial neural network (ANN, default: FireNet) or a spiking neural network (SNN, default: LIF-FireNet), respectively. In `configs/`, you can find the aforementioned configuration files and vary the training settings (e.g., model, number of input events, activate/deactivate visualization). For other models available, see `models/model.py`. 

**Note that we used a batch size of 8 in our experiments. Depending on your computational resources, you may need to lower this number.**

During and after the training, information about your run can be visualized through MLflow.

## Uninstalling pyenv

Once you finish using our code, you can uninstall `pyenv` from your system by:

1. Removing the `pyenv` configuration lines from your `~/.bashrc`.
2. Removing its root directory. This will delete all Python versions that were installed under the `$HOME/.pyenv/versions/` directory:

```
rm -rf $HOME/.pyenv/
```
