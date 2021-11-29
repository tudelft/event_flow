import numpy as np
import torch
import yaml


class YAMLParser:
    """ YAML parser for optical flow and image reconstruction config files """

    def __init__(self, config):
        self.reset_config()
        self.parse_config(config)
        self.get_device()
        self.init_seeds()

    def parse_config(self, file):
        with open(file) as fid:
            yaml_config = yaml.load(fid, Loader=yaml.FullLoader)
        self.parse_dict(yaml_config)

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return self._device

    @property
    def loader_kwargs(self):
        return self._loader_kwargs

    def reset_config(self):
        self._config = {}

        # MLFlow experiment name
        self._config["experiment"] = "Default"

        # input data mode
        self._config["data"] = {}
        self._config["data"]["mode"] = "events"
        self._config["data"]["window"] = 5000

        # data loader
        self._config["loader"] = {}
        self._config["loader"]["resolution"] = [180, 240]
        self._config["loader"]["batch_size"] = 1
        self._config["loader"]["augment"] = []
        self._config["loader"]["gpu"] = 0
        self._config["loader"]["seed"] = 0

        # hot pixel
        self._config["hot_filter"] = {}
        self._config["hot_filter"]["enabled"] = True
        self._config["hot_filter"]["max_px"] = 100
        self._config["hot_filter"]["min_obvs"] = 5
        self._config["hot_filter"]["max_rate"] = 0.8

        # model
        self._config["model"] = {}

        # neuron
        self._config["spiking_neuron"] = {}

        # visualization
        self._config["vis"] = {}
        self._config["vis"]["bars"] = False

    def update(self, config):
        self.reset_config()
        self.parse_config(config)

    def parse_dict(self, input_dict, parent=None):
        if parent is None:
            parent = self._config
        for key, val in input_dict.items():
            if isinstance(val, dict):
                if key not in parent.keys():
                    parent[key] = {}
                self.parse_dict(val, parent[key])
            else:
                parent[key] = val

    def get_device(self):
        cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:" + str(self._config["loader"]["gpu"]) if cuda else "cpu")
        self._loader_kwargs = {"num_workers": 0, "pin_memory": True} if cuda else {}

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def init_seeds(self):
        torch.manual_seed(self._config["loader"]["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._config["loader"]["seed"])
            torch.cuda.manual_seed_all(self._config["loader"]["seed"])

    def merge_configs(self, run):
        """
        Overwrites mlflow metadata with configs.
        """

        # parse mlflow settings
        config = {}
        for key in run.keys():
            if len(run[key]) > 0 and run[key][0] == "{":  # assume dictionary
                config[key] = eval(run[key])
            else:  # string
                config[key] = run[key]

        # overwrite with config settings
        self.parse_dict(self._config, config)
        self.combine_entries(config)

        return config

    @staticmethod
    def combine_entries(config):
        """
        Combines entries that had to be split because of MLFlow's max character limit.
        """

        if "spiking_neuron" in config.keys():
            config["model"]["spiking_neuron"] = config["spiking_neuron"]
            config.pop("spiking_neuron", None)

        return config
