from src.utils.model_handling import load_model, get_model_dataset, update_state_action_buffers, get_model_device, init_state_action_buffers, get_models_evaluation_data, get_model_cfg_from_checkpoint, load_checkpoint
from src.utils.gym_env_handling import save_animation
from src.utils.json_handling import export_config_2_json_file
from src.utils.gym_env_handling import sample_env_setting
import torch, os, pickle
import gymnasium as gym
import numpy as np
import warnings
from tqdm import tqdm
import shutil

def test(test_cfg, model_dir:str=None):
    """
    """
    model_dir = 'models' if model_dir is None else model_dir
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    models_path = os.path.join(project_root, 'results', model_dir)

    rewards = None

    if isinstance(test_cfg.model, str):
        if len(test_cfg.model):
            # Single model to be tested.
            rewards = test_model_on_multiple_envs(
                model_id=test_cfg.model,
                nb_test_episodes=test_cfg.nb_test_episodes,
                reward_per_episode=test_cfg.reward_per_episode,
                export_animation=test_cfg.save_animation,
                model_dir=model_dir,
                sequence_length=test_cfg.sequence_length,
                env_coef_of_var=test_cfg.env_coef_of_var,
                seed_coef_of_var=test_cfg.seed_coef_of_var
            )

            # Export to file.
            if test_cfg.export_to_file:
                export_rewards_2_file(model_id=test_cfg.model, rewards=rewards, reward_per_episode=test_cfg.reward_per_episode, model_dir=model_dir)
                export_config_2_json_file(config=test_cfg, file_name=f'test_config-{test_cfg.model}', path=os.path.join(models_path, test_cfg.model))
        else:
            # Empty model id - test all models found in `/results/models/`.
            models_list = os.listdir(models_path)

            if len(models_list):
                rewards = []
                for model_id in models_list:
                    rewards.append(
                        test_model_on_multiple_envs(
                            model_id=model_id,
                            nb_test_episodes=test_cfg.nb_test_episodes,
                            reward_per_episode=test_cfg.reward_per_episode,
                            export_animation=test_cfg.save_animation,
                            model_dir=model_dir,
                            sequence_length=test_cfg.sequence_length,
                            env_coef_of_var=test_cfg.env_coef_of_var,
                            seed_coef_of_var=test_cfg.seed_coef_of_var
                        )
                    )
                    # Export to file.
                    if test_cfg.export_to_file:
                        export_rewards_2_file(model_id=model_id, rewards=rewards[-1], reward_per_episode=test_cfg.reward_per_episode, model_dir=model_dir)
                        export_config_2_json_file(config=test_cfg, file_name=f'test_config-{model_id}', path=os.path.join(models_path, model_id))
            else:
                warnings.warn(f'No models found in {models_path}.')

    elif isinstance(test_cfg.model, list):
        # Multiple model IDs in a list to be tested.
        if len(test_cfg.model):
            rewards = []
            
            for model_id in test_cfg.model:
                rewards.append(
                    test_model_on_multiple_envs(
                        model_id=model_id,
                        nb_test_episodes=test_cfg.nb_test_episodes,
                        reward_per_episode=test_cfg.reward_per_episode,
                        export_animation=test_cfg.save_animation,
                        model_dir=model_dir,
                        sequence_length=test_cfg.sequence_length,
                        env_coef_of_var=test_cfg.env_coef_of_var,
                        seed_coef_of_var=test_cfg.seed_coef_of_var
                    )
                )
                # Export to file.
                if test_cfg.export_to_file:
                    export_rewards_2_file(model_id=model_id, rewards=rewards[-1], reward_per_episode=test_cfg.reward_per_episode, model_dir=model_dir)
                    export_config_2_json_file(config=test_cfg, file_name=f'test_config-{model_id}', path=os.path.join(models_path, model_id))
        else:
            # Empty list - test all models found in `/results/models/`.
            models_list = os.listdir(models_path)

            if len(models_list):
                rewards = []
                for model_id in models_list:
                    rewards.append(
                        test_model_on_multiple_envs(
                            model_id=model_id,
                            nb_test_episodes=test_cfg.nb_test_episodes,
                            reward_per_episode=test_cfg.reward_per_episode,
                            export_animation=test_cfg.save_animation,
                            model_dir=model_dir,
                            sequence_length=test_cfg.sequence_length,
                            env_coef_of_var=test_cfg.env_coef_of_var,
                            seed_coef_of_var=test_cfg.seed_coef_of_var
                        )
                    )
                    # Export to file.
                    if test_cfg.export_to_file:
                        export_rewards_2_file(model_id=model_id, rewards=rewards[-1], reward_per_episode=test_cfg.reward_per_episode, model_dir=model_dir)
                        export_config_2_json_file(config=test_cfg, file_name=f'test_config-{model_id}', path=os.path.join(models_path, model_id))
            else:
                warnings.warn(f'No models found in {models_path}.')

    else:
        pass

    return rewards

def test_model_on_multiple_envs(
        model_id:str,
        nb_test_episodes:int,
        reward_per_episode:str,
        export_animation:bool=False,
        overrride_existing:bool=True,
        model_dir:str=None,
        sequence_length:int=-1,
        env_coef_of_var:float=0,
        seed_coef_of_var:float=0
    ) -> dict:
    """
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_dir = 'models' if model_dir is None else model_dir

    # Instantiate model
    # -------------------------------------------
    model = load_model(model_id=model_id, model_dir_name=model_dir)
    device = get_model_device(model_id=model_id, model_dir_name=model_dir)

    # Load model's config
    # -------------------------------------------
    chkpt = load_checkpoint(model_id=model_id, model_dir_name=model_dir)
    model_cfg = get_model_cfg_from_checkpoint(checkpoint=chkpt)

    # Loading training dataset (prevent data leakage)
    # -------------------------------------------
    dataset_name = get_model_dataset(model_id=model_id, model_dir_name=model_dir) # dataset (.pt) file used for training.
    dataset = torch.load(os.path.join(project_root, 'data', 'processed', dataset_name), weights_only=False)

    # Override existing animations
    # -------------------------------------------
    gif_dir = f'animations-sequence_length_{model_cfg.training_seq_len if sequence_length == -1 else sequence_length}'
    gif_dir += f'-env_setup-coef_of_var_{env_coef_of_var}'
    gif_dir += f'_seed_{seed_coef_of_var}' if env_coef_of_var != 0 else ''
    gif_path = os.path.join(project_root, 'results', model_dir, model_id, gif_dir)

    if overrride_existing and os.path.exists(gif_path) and os.path.isdir(gif_path):
        shutil.rmtree(gif_path)

    # Simulate env
    # -------------------------------------------
    test_iter = range(dataset.nb_episodes, dataset.nb_episodes+nb_test_episodes)
    progress = tqdm(test_iter, desc=f'Testing model ID {model_id}', unit='env(seed)')
    
    rewards_per_env_seed = {}
    for rand_seed in progress:

        # play env
        frames, reward_per_step, _, _, _ = play_env(
            device=device,
            rand_seed=rand_seed,
            model=model,
            dataset=dataset,
            sequence_length=model_cfg.training_seq_len if sequence_length == -1 else sequence_length,
            env_coef_of_var=env_coef_of_var,
            seed_coef_of_var=seed_coef_of_var
        )
        
        rewards_per_env_seed[f'env_seed_{rand_seed}'] = np.sum(reward_per_step) if reward_per_episode == 'accumulated' else np.array(reward_per_step)

        progress.set_postfix({'accumulated reward': f'{np.sum(reward_per_step):.3f}'})

        if export_animation:
            save_animation(
                frames=frames, path=gif_path, file_name=f'env_seed_{rand_seed}_{int(np.sum(reward_per_step))}.gif'
            )

    return rewards_per_env_seed

def play_env(device, rand_seed, model, dataset, sequence_length, env_coef_of_var, seed_coef_of_var):

    if env_coef_of_var == 0:
            env = gym.make("LunarLander-v3", render_mode="rgb_array")
    else:
        (gravity, wind_power, turbulence_power) = sample_env_setting(
            coef_var=env_coef_of_var,
            seed=seed_coef_of_var
        )

        env = gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
            enable_wind=True,
            gravity=gravity,
            wind_power=wind_power,
            turbulence_power=turbulence_power
        )

    state, info = env.reset(seed=rand_seed)
    done = False
    reward_per_step = []
    states = [state]

    # ---- Create State/Action Buffers ----
    states_buffer, actions_buffer = init_state_action_buffers(
        state=state, dataset_mean=dataset.mean, dataset_std=dataset.std, normalize=dataset.normalized
    )

    # ---- Play Env. ----
    frames = []
    model.eval()
    while not done:
        # -- Cast input. --
        states_seq = torch.tensor(states_buffer, dtype=torch.float32).unsqueeze(0).to(device)
        actions_seq = torch.tensor(actions_buffer, dtype=torch.long).unsqueeze(0).to(device)

        padding_mask = torch.ones((states_buffer.shape[0]+actions_buffer.shape[0],), dtype=torch.long).unsqueeze(0).to(device)

        # -- Get action from the model. --
        with torch.no_grad():
            # print(states_buffer.shape, actions_buffer.shape)
            logits = model(states_seq=states_seq, actions_seq=actions_seq, padding_mask=padding_mask)[0, :]
            action = torch.argmax(logits).item()

        # -- Step the environment. --
        state, reward, terminated, truncated, info = env.step(action)
        reward_per_step.append(reward)
        
        # -- Update stacked states. --
        states_buffer, actions_buffer = update_state_action_buffers(
            max_len=sequence_length,
            next_state=state,
            next_action=action,
            states_buffer=states_buffer,
            actions_buffer=actions_buffer,
            mean=dataset.mean,
            std=dataset.std,
            normalize=dataset.normalized
        )

        # -- Update env. --
        frame = env.render() 
        frames.append(frame)
        done = terminated or truncated

        states.append(state)

    env.close()

    return frames, reward_per_step, states, terminated, truncated

def export_rewards_2_file(model_id:str, rewards:dict, reward_per_episode:str, model_dir:str=None) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_dir = 'models' if model_dir is None else model_dir
    models_path = os.path.join(project_root, 'results', model_dir, model_id)

    with open(os.path.join(models_path, f'evaluation-{model_id}-reward_per_episode_{reward_per_episode}.pkl'), 'wb') as file:
        pickle.dump(rewards, file)

def test_hpo(test_cfg, model, train_dataset):
    from src.hpo.custom_fitness_functions import evaluate_fitness_with_AUC

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    
    # Simulate env
    # -------------------------------------------
    test_iter = range(train_dataset.nb_episodes, train_dataset.nb_episodes+test_cfg.nb_test_episodes)
    progress = tqdm(test_iter, desc=f'Testing model ID {test_cfg.model_id}', unit='env(seed)')

    fitnesses = []
    for rand_seed in progress:

        # play env
        frames, reward_per_step, states, _, _ = play_env(test_cfg.device, rand_seed, model, train_dataset, None, test_cfg.sequence_length)
        
        fitnesses.append(evaluate_fitness_with_AUC(states[-1], reward_per_step))

        progress.set_postfix({'accumulated reward': f'{np.sum(reward_per_step):.3f}'})

    return fitnesses