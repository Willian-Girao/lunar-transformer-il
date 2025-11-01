import numpy as np
from tqdm import tqdm
import torch, os
import random
import string
from src.utils.model_handling import get_param_sum

def train_model(train_cfg, model_cfg, model, optimizer, criterion, train_dataloader, show_param_sum=True, epoch_eval:bool=False, eval_dataloader=None) -> str:
    model_id = generate_checkpoint_id()
    epochs_losses = []
    epoch_iter = range(train_cfg.epochs)

    param_sum = get_param_sum(model=model)
    show = f' (param. sum {param_sum:.3f})' if show_param_sum else ''

    if train_cfg.progress:
        epoch_iter = tqdm(epoch_iter, desc=f'Training model ID {model_id} ({model_cfg.depth} {model_cfg.num_attention_heads} {model_cfg.embedding_dim} {model_cfg.intermediate_dim}) seed {model_cfg.seed} {show}', unit='epoch')

    model.train()
    for ep in epoch_iter:
        epoch_loss = []

        for ith_batch, (states_batch, actions_batch, labels_batch, masks_batch, reward) in enumerate(train_dataloader):
            states_batch = states_batch.to(train_cfg.device)
            actions_batch = actions_batch.to(train_cfg.device)
            labels_batch = labels_batch.to(train_cfg.device)
            masks_batch = masks_batch.to(train_cfg.device)
            reward = reward.to(train_cfg.device)

            optimizer.zero_grad()
            logits = model(states_seq=states_batch, actions_seq=actions_batch, padding_mask=masks_batch, noise_seed=ith_batch)
            loss = criterion(logits, labels_batch)
            loss = (loss * reward).mean()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        epochs_losses.append(np.mean(epoch_loss))

        if epoch_eval:
            eval_losses, correct_rate = eval_model(model, eval_dataloader, criterion, train_cfg)
            create_model_checkpoint(model_id, model, model_cfg, train_cfg, optimizer, epochs_losses, eval_losses, correct_rate)

            if train_cfg.progress:
                epoch_iter.set_postfix({'loss': f'{epochs_losses[-1]:.10f}', 'CR': f'{correct_rate:.2f}'})
        else:
            create_model_checkpoint(model_id, model, model_cfg, train_cfg, optimizer, epochs_losses)

            if train_cfg.progress:
                epoch_iter.set_postfix({'loss': f'{epochs_losses[-1]:.10f}'})

    return model_id

def eval_model(model, eval_dataloader, criterion, train_cfg):
    size = len(eval_dataloader.dataset)
    num_batches = len(eval_dataloader)

    eval_losses, correct_rate = 0, 0

    model.eval()
    with torch.no_grad():
        for ith_batch, (states_batch, actions_batch, labels_batch, masks_batch, _) in enumerate(eval_dataloader):
            states_batch = states_batch.to(train_cfg.device)
            actions_batch = actions_batch.to(train_cfg.device)
            labels_batch = labels_batch.to(train_cfg.device)
            masks_batch = masks_batch.to(train_cfg.device)

            logits = model(states_seq=states_batch, actions_seq=actions_batch, padding_mask=masks_batch, noise_seed=ith_batch)

            eval_losses += criterion(logits, labels_batch).mean().item()

            predictions = torch.argmax(logits, dim=-1)
            correct_rate += (predictions == labels_batch).type(torch.float).sum().item()

    eval_losses /= num_batches
    correct_rate /= size

    return eval_losses, float(correct_rate)

def create_model_checkpoint(model_id, model, model_cfg, train_cfg, optimizer, epochs_losses, eval_losses=None, correct_rate=None):
    """
    Saves a model checkpoint containing:
    - model weights
    - optimizer state
    - model and training configurations

    Args:
     model: The trained model.
     model_cfg: TransformerConfig instance.
     train_cfg: TransformerConfig or dict with training hyperparameters.
     optimizer: Optimizer instance.
     filename (str): Name of the checkpoint file.
    """
    from src.utils.model_handling import save_model

    # Convert configs safely to standard dicts
    model_cfg_dict = dict(model_cfg)
    training_cfg_dict = dict(train_cfg)

    # Build checkpoint dictionary
    checkpoint = {
        **model_cfg_dict,
        **training_cfg_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'epochs_losses': np.array(epochs_losses),
    }

    if eval_losses is not None and correct_rate is not None:
        checkpoint.update({
            'eval_losses': eval_losses,
            'correct_rate': correct_rate
        })

    # Save checkpoint
    save_model(
        checkpoint=checkpoint,
        model_id=model_id,
        file_name=f'{model_id}-{model_cfg.training_seq_len}-{model_cfg.name}-{train_cfg.epochs}.pth'
    )

def generate_checkpoint_id(digs:int=3, chrs:int=2) -> str:
    """
    """
    random.seed()
    
    digits = [random.choice(string.digits) for _ in range(digs)]
    letters = [random.choice(string.ascii_lowercase) for _ in range(chrs)]
    
    model_id_list = digits + letters
    random.shuffle(model_id_list)
    
    return ''.join(model_id_list)