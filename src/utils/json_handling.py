import os
import json
import torch
from typing import Union

def json_2_dict(json_file: str) -> dict:
    """ Transforms a .json file into a dictionary.

    This does not support nested keys.

    Args:
        json_file (str): The name of the .json file to be opened.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    json_path = os.path.join(project_root, 'configs', json_file)

    with open(json_path) as f:
        json_config = json.load(f)

    dict_ = {}
    for key in json_config.keys():
        dict_[key] = json_config[key]

    return dict_

def json_2_nni_search_space_dict(json_file: str) -> dict:
    """ Transforms a .json file into a dictionary for NNI.

    The `json_file` will define the search space for NNI. The json needs
    to have lists as its values, where the 1st position is a string defining
    the type of the search space (e.g., 'choice', 'uniform') and the 2nd position
    is another list of two values defining the lower/upper range of values.

    This does not support nested keys.

    Args:
        json_file (str): The name of the .json file to be opened.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    json_path = os.path.join(project_root, 'configs', json_file)

    with open(json_path) as f:
        json_config = json.load(f)

    dict_ = {}
    for key in json_config.keys():
        dict_[key] = {'_type': json_config[key][0], '_value': json_config[key][1]}

    return dict_

def export_config_2_json_file(config:dict, file_name:str, path:os.path) -> None:
    """ Exports a dictionary to a .json file.

    This does not support nested keys.

    Args:
        config (dict): The dictionary to be turned into a .json.
        file_name (str): The name for the .json file.
        path (os.path): Location where the file will be saved.
    """
    serializable_config = make_serializable(config)
    with open(os.path.join(path, f'{file_name}.json'), 'w') as fp:
        json.dump(serializable_config, fp)

def make_serializable(obj) -> Union[str, list, dict]:
    """ Serializes a data structure.
    """
    if isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj