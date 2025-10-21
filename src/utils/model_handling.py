import os, torch
from pathlib import Path
import warnings
import numpy as np
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

def load_model(model_id:str) -> DecoderTransformer:
    """
    """
    # Load checkpoint
    # -------------------------------------------
    checkpoint = load_checkpoint(model_id=model_id)

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
            raise ValueError(f'Key `{key}` could not be found within the models checkpoint `.pth` file.')
    
    return model_cfg

def load_checkpoint(model_id:str):
    """
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_dir = os.path.join(project_root, 'results', 'models', model_id)
    pth_files = sorted(list(Path(model_dir).rglob('*.pth')))

    if not len(pth_files):
        raise ValueError(f'No .pth files in {model_dir}.')
    
    if len(pth_files) > 1:
        raise warnings.warn(f'{len(pth_files)} .pth files found in {os.path.join(project_root, model_id)}. The 1st element in the list will be used.')
    
    checkpoint = torch.load(os.path.join(model_dir, pth_files[0]), weights_only=False)

    return checkpoint

def get_model_dataset(model_id:str) -> str:
    # Load checkpoint
    # -------------------------------------------
    checkpoint = load_checkpoint(model_id=model_id)

    if 'dataset' not in checkpoint:
        raise ValueError(f"Can not find dataset name inside model's checkpoint.")

    return checkpoint['dataset']

def get_model_device(model_id:str) -> str:
    # Load checkpoint
    # -------------------------------------------
    checkpoint = load_checkpoint(model_id=model_id)

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