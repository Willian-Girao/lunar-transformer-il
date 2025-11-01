import os
from datetime import datetime
from src.utils.model_handling import load_checkpoint, get_model_cfg_from_checkpoint, get_training_cfg_from_checkpoint, get_models_evaluation_data
from src.utils.evaluation_handling import get_top_n_rewards

def log_evaluated_model(model_id:str, model_dir_name:str=None) -> None:
    """
    """
    model_chkpt = load_checkpoint(model_id=model_id, model_dir_name=model_dir_name)

    model_cfg = get_model_cfg_from_checkpoint(checkpoint=model_chkpt)
    training_cfg = get_training_cfg_from_checkpoint(checkpoint=model_chkpt)
    eval_data = get_models_evaluation_data(model_id=model_id, model_dir=model_dir_name)
    top_n = get_top_n_rewards(eval_data=eval_data)

    content_2_log = {**model_cfg, **training_cfg, **top_n}
    content_2_log['model location'] = model_dir_name

    log_2_txt(content_2_log=content_2_log, model_id=model_id)

def log_2_txt(content_2_log: dict, model_id: str) -> None:
    """
    Logs model evaluation content to a text file.

    Args:
        content_2_log (dict): Dictionary containing model, training, and evaluation info.
        model_id (str): Model identifier used for naming the log file.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    logs_dir = os.path.join(project_root, 'results', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, f'{model_id}_log.txt')

    # Prepare timestamp and log header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'='*80}\nMODEL LOG - {model_id}\nTimestamp: {timestamp}\n{'='*80}\n"

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(header)
        for key, value in content_2_log.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")