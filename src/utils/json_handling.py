import os
import json
import torch

def json_2_dict(json_file: str) -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    json_path = os.path.join(project_root, 'configs', json_file)

    with open(json_path) as f:
        json_config = json.load(f)

    dict_ = {}
    for key in json_config.keys():
        dict_[key] = json_config[key]

    return dict_

def json_2_nni_search_space_dict(json_file: str) -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    json_path = os.path.join(project_root, 'configs', json_file)

    with open(json_path) as f:
        json_config = json.load(f)

    dict_ = {}
    for key in json_config.keys():
        dict_[key] = {'_type': json_config[key][0], '_value': json_config[key][1]}

    return dict_

def export_config_2_json_file(config:dict, file_name:str, path:os.path):
    serializable_config = make_serializable(config)
    with open(os.path.join(path, f'{file_name}.json'), 'w') as fp:
        json.dump(serializable_config, fp)

def make_serializable(obj):
    if isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj