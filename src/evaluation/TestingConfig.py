import torch
import os
import json

class TestingConfig(dict):
    """ A dictionary subclass that allows both dot notation and
    key-based access.
    """
    defaults = {
        'seed': 0,
        'model': [], 
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    def __init__(self, **kwargs):
        super().__init__(self.defaults) # default attributes
        self.update(kwargs)             # use-provided ones

    # Allow accessing dictionary items as attributes, e.g. cfg.key
    # When you do cfg.key, Python calls cfg.__getattr__("key"), which now does cfg.get("key")
    __getattr__ = dict.get

    # Allow setting dictionary items as attributes, e.g. cfg.key = value
    # When you do cfg.key = value, Python calls cfg.__setattr__("key", value), which now does cfg["key"] = value
    __setattr__ = dict.__setitem__

    # Allow deleting dictionary items as attributes, e.g. del cfg.key
    # When you do del cfg.key, Python calls cfg.__delattr__("key"), which now does del cfg["key"]
    __delattr__ = dict.__delitem__

    def from_json(self, json_file: os.path) -> None:
        """ Loads a .json file containing the configuration necessary to set
        the classe's default attributes.

        Args:
         json_file: Path to a .json file containing the necessary to set the
          attributes of an instance of `TrainingConfig`.
        """
        with open(json_file) as f:
            json_config = json.load(f)

        # Loop over the attributes of `TrainingConfig` and try to look them up
        # within `json_config`.
        for key in json_config.keys():
            self[key] = json_config[key]

    def from_dict(self, dict: dict) -> None:
        """
        """
        for key in dict.keys():
            self[key] = dict[key]     