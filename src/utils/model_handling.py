import os, torch, pickle
from pathlib import Path
import warnings
import numpy as np
import torch.nn as nn
from src.models.decoder_only_transformer.DecoderTransformer import DecoderTransformer

def save_model(checkpoint:dict, model_id:str, file_name:str) -> None:
    """
    """
    # Create export folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    save_dir = os.path.join(project_root, 'results', 'models', model_id)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, file_name)

    # Save checkpoint
    torch.save(checkpoint, model_path)

def load_model(model_id:str, model_dir_name:str=None) -> DecoderTransformer:
    """
    """
    # Load checkpoint
    # -------------------------------------------
    model_dir_name = 'models' if model_dir_name is None else model_dir_name
    checkpoint = load_checkpoint(model_id=model_id, model_dir_name=model_dir_name)

    # Instantiate model
    # -------------------------------------------
    model_cfg = get_model_cfg_from_checkpoint(checkpoint=checkpoint)

    model = DecoderTransformer(model_cfg).to(checkpoint['device'])
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def get_model_cfg_from_checkpoint(checkpoint):
    """
    """
    from src.models.decoder_only_transformer.TransformerConfig import TransformerConfig

    model_cfg = TransformerConfig()

    for key in model_cfg.keys():
        if key in checkpoint:
            model_cfg[key] = checkpoint[key]
        else:
            raise ValueError(f'Key `{key}` could not be found within the model`s configuration object.')
    
    return model_cfg

def get_training_cfg_from_checkpoint(checkpoint):
    """
    """
    from src.training.TrainingConfig import TrainingConfig

    training_cfg = TrainingConfig()

    for key in training_cfg.keys():
        if key in checkpoint:
            training_cfg[key] = checkpoint[key]
        else:
            raise ValueError(f'Key `{key}` could not be found within the model`s training configuration object.')
    
    return training_cfg

def load_checkpoint(model_id:str, model_dir_name:str=None):
    """
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    if model_dir_name is not None:
        model_dir = os.path.join(project_root, 'results', model_dir_name, model_id)
    else:
        model_dir = os.path.join(project_root, 'results', 'models', model_id)

    pth_files = sorted(list(Path(model_dir).rglob('*.pth')))

    if not len(pth_files):
        raise ValueError(f'No .pth files in {model_dir}.')
    
    if len(pth_files) > 1:
        raise warnings.warn(f'{len(pth_files)} .pth files found in {os.path.join(project_root, model_id)}. The 1st element in the list will be used.')
    
    checkpoint = torch.load(os.path.join(model_dir, pth_files[0]), weights_only=False)

    return checkpoint

def get_model_dataset(model_id:str, model_dir_name:str=None) -> str:
    # Load checkpoint
    # -------------------------------------------
    model_dir_name = 'models' if model_dir_name is None else model_dir_name
    checkpoint = load_checkpoint(model_id=model_id, model_dir_name=model_dir_name)

    if 'dataset' not in checkpoint:
        raise ValueError(f"Can not find dataset name inside model's checkpoint.")

    return checkpoint['dataset']

def get_model_device(model_id:str, model_dir_name:str=None) -> str:
    # Load checkpoint
    # -------------------------------------------
    model_dir_name = 'models' if model_dir_name is None else model_dir_name
    checkpoint = load_checkpoint(model_id=model_id, model_dir_name=model_dir_name)

    if 'device' not in checkpoint:
        raise ValueError(f"Can not find device name inside model's checkpoint.")

    return checkpoint['device']

def update_state_action_buffers(max_len, next_state, next_action, states_buffer, actions_buffer, mean, std, normalize: bool):
    if normalize:
        next_state[:6] = (next_state[:6] - mean) / std # normalize
    
    if states_buffer.shape[0] >= max_len//2:
        states_buffer = np.delete(states_buffer, 0, 0) # remove oldest state
    states_buffer = np.append(states_buffer, np.array([next_state]), axis=0) # insert current state

    if actions_buffer.shape[0] >= max_len//2:
        actions_buffer = np.delete(actions_buffer, 0, 0) # remove oldest action
    actions_buffer = np.append(actions_buffer, np.array([next_action]), axis=0) # insert current action

    return states_buffer, actions_buffer

def init_state_action_buffers(state, dataset_mean, dataset_std, normalize):
    states_buffer = np.array([
        state for _ in range(1)
    ])

    actions_buffer = np.array([
        0 for _ in range(1)
    ])

    # normalize based on mean/std computed during training
    if normalize:
        states_buffer[..., :6] = (states_buffer[..., :6]  - dataset_mean) / dataset_std

    return states_buffer, actions_buffer

def get_param_sum(model: nn.Module):
    total_sum = 0.0
    for param in model.parameters():
        total_sum += param.data.sum().item()
    return total_sum

def get_models_training_loss(model_id:str):
    checkpoint = load_checkpoint(model_id)
    return checkpoint['epochs_losses']

def get_model_training_metrics(model_id:str, model_dir:str=None):
    model_dir = 'models' if model_dir is None else model_dir
    checkpoint = load_checkpoint(model_id, model_dir)

    epochs_losses = checkpoint['epochs_losses']

    eval_losses = None
    if 'eval_losses' in checkpoint:
        eval_losses = checkpoint['eval_losses']

    correct_rate = None
    if 'correct_rate' in checkpoint:
        correct_rate = checkpoint['correct_rate']
    
    return epochs_losses, eval_losses, correct_rate

def get_models_evaluation_data(model_id:str, model_dir:str=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_path = os.path.join(
        project_root,
        'results',
        'models' if model_dir is None else model_dir,
        model_id
    )
    files = os.listdir(model_path)
    
    matching_file = None
    for f in files:
        if f.startswith(f"evaluation-{model_id}") and f.endswith(".pkl"):
            matching_file = f
            break

    if matching_file is None:
        return None
    
    file_path = os.path.join(model_path, matching_file)
    with open(file_path, 'rb') as f:
        eval_data = pickle.load(f)
    
    return eval_data