import torch
import os
import json

class TransformerConfig(dict):
    """ A dictionary subclass that allows both dot notation and
    key-based access.
    """
    defaults = {
        'nb_actions': 4,
        'state_space_dim': 8,
        'training_seq_len': 4,      # smallest allowed by the decoder transformer.
        'embedding_dim': 0,
        'token_types': 2,           # state vector or acion taken
        'num_attention_heads': 0,
        'intermediate_dim': 0,
        'hidden_dropout_prob': 0,
        'depth': 0,
        'seed': 0,
        'noise_type': 'none',
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
          attributes of an instance of `TransformerConfig`.
        """
        with open(json_file) as f:
            json_config = json.load(f)

        # Loop over the attributes of `TransformerConfig` and try to look them up
        # within `json_config`.
        for key in json_config.keys():
            self[key] = json_config[key]                