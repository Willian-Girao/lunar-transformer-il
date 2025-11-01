from IPython.display import Image, display_html
import ipywidgets as widgets
from src.utils.model_handling import load_checkpoint, get_models_training_loss
import time, os, pickle
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import numpy as np

def show_gifs_in_row(model_id: str, width: int = 500, output_widget=None, models_dir_name:str=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    if models_dir_name is not None:
        results_root = os.path.join(project_root, 'results', models_dir_name)
    else:
        results_root = os.path.join(project_root, 'results', 'models')

    path = os.path.join(results_root, model_id, 'animations')

    gifs = [f for f in os.listdir(path) if f.lower().endswith('.gif')]
    if not gifs:
        print("No .gif files found in the provided path.")
        return

    model_check = load_checkpoint(model_id=model_id, model_dir_name=models_dir_name)
    unique_tag = str(time.time())
    html = "".join(
        f"<div style='display:inline-block; text-align:center; margin:5px;'>"
        f"<p style='font-size:16px; margin-bottom:4px;'>env seed {gif.split('_')[2]} | model seed {model_check['seed']} | reward {gif.split('_')[-1].replace('.gif', '')}</p>"
        f"<img src='{os.path.relpath(os.path.join(path, gif), start=os.getcwd())}?{unique_tag}' "
        f"style='width:{width}px; display:block; margin:auto;'/>"
        f"</div>"
        for gif in gifs
    )

    if output_widget:
        output_widget.clear_output(wait=True)
        with output_widget:
            display(HTML(html))
    else:
        display(HTML(html))

def create_gif_dropdown(models_dir_name:str=None):
    output = widgets.Output()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    if models_dir_name is not None:
        results_root = os.path.join(project_root, 'results', models_dir_name)
    else:
        results_root = os.path.join(project_root, 'results', 'models')

    names = sorted(os.listdir(results_root))

    dropdown = widgets.Dropdown(
        options=["Select a model..."] + names,
        value="Select a model...",
        description='Model:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='250px')
    )

    # Only attach callback if not already attached
    if not hasattr(dropdown, "_custom_callback_attached"):
        def on_select(change):
            selected_name = change['new']
            if selected_name == "Select a model...":
                return
            show_gifs_in_row(model_id=selected_name, output_widget=output, models_dir_name=models_dir_name)

        dropdown.observe(on_select, names='value')
        dropdown._custom_callback_attached = True  # mark as attached

    display(widgets.VBox([dropdown, output]))

def display_training_loss(model_id: str, ylim=(0, 0.1)):
    """Plot training loss."""
    model_check = load_checkpoint(model_id=model_id)
    plt.figure(figsize=(6, 3))
    plt.plot(model_check['epochs_losses'], label="Training Loss", color='blue')
    plt.title(f"Training Loss — {model_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(ylim)
    plt.xlim(0, len(model_check['epochs_losses']))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def get_all_rewards(models_dir_name:str=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    if models_dir_name is not None:
        results_root = os.path.join(project_root, 'results', models_dir_name)
    else:
        results_root = os.path.join(project_root, 'results', 'models')

    models = sorted(os.listdir(results_root))

    env_seeds = []
    rewards = []

    for model_id in models:
        eval_file = f'evaluation-{model_id}-reward_per_episode_total.pkl'
        try:
            with open(os.path.join(results_root, model_id, eval_file), 'rb') as file:
                eval_data = pickle.load(file)

            for env_seed, reward in eval_data.items():
                env_seeds.append(int(env_seed.split('_')[-1]))
                rewards.append(reward)

        except:
            print(f'skipping {model_id}: {eval_file} does not exist.', end='\r', flush=True)
            pass

    return rewards, env_seeds

def plot_all_losses(cap_epoch: int = 0, tag=[]):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_root = os.path.join(project_root, 'results', 'models')
    models = sorted(os.listdir(results_root))

    plt.figure(figsize=(20, 10))

    for model_id in models:
        training_loss = get_models_training_loss(model_id)
        if len(training_loss) > cap_epoch:
            print(model_id)
        # cap epoch if needed
        if cap_epoch > 0:
            training_loss = training_loss[:cap_epoch]

        # styling
        alpha = 0.75 if model_id in tag else 0.5
        ls = '-' if model_id in tag else '-.'
        lw = 1.5 if model_id in tag else 0.8

        plt.plot(training_loss, label=model_id, alpha=alpha, ls=ls, lw=lw)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')

    # Legend outside, on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.tight_layout()  # adjust to make room for legend
    plt.show()


def plot_rewards_tep(reward_per_step):
    # Assuming reward_per_step is a list or numpy array
    rewards = np.array(reward_per_step[:-10])
    accumulated = np.cumsum(rewards)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Left y-axis: reward per step
    ax1.plot(rewards, color='tab:blue', label='Reward per step', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward per step', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Right y-axis: accumulated reward
    ax2 = ax1.twinx()
    ax2.plot(accumulated, color='tab:orange', label='Accumulated reward', lw=1.5)
    ax2.set_ylabel('Accumulated reward', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Reward per Step and Accumulated Reward')
    plt.tight_layout()
    plt.show()