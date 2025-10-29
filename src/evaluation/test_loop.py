from src.utils.model_handling import load_model, get_model_dataset, update_state_action_buffers, get_model_device, init_state_action_buffers
from src.utils.gym_env_handling import save_animation
from src.utils.json_handling import export_config_2_json_file
import torch, os, pickle
import gymnasium as gym
import numpy as np
import warnings
from tqdm import tqdm

def test(test_cfg):
    """
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    models_path = os.path.join(project_root, 'results', 'models')

    rewards = None

    if isinstance(test_cfg.model, str):
        if len(test_cfg.model):
            # Single model to be tested.
            rewards = test_model_on_multiple_envs(
                model_id=test_cfg.model,
                nb_test_episodes=test_cfg.nb_test_episodes,
                sequence_length=test_cfg.sequence_length,
                reward_per_episode=test_cfg.reward_per_episode,
                export_animation=test_cfg.save_animation
            )

            # Export to file.
            if test_cfg.export_to_file:
                export_rewards_2_file(model_id=test_cfg.model, rewards=rewards, reward_per_episode=test_cfg.reward_per_episode)
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
                            sequence_length=test_cfg.sequence_length,
                            reward_per_episode=test_cfg.reward_per_episode,
                            export_animation=test_cfg.save_animation
                        )
                    )
                    # Export to file.
                    if test_cfg.export_to_file:
                        export_rewards_2_file(model_id=model_id, rewards=rewards[-1], reward_per_episode=test_cfg.reward_per_episode)
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
                        sequence_length=test_cfg.sequence_length,
                        reward_per_episode=test_cfg.reward_per_episode,
                        export_animation=test_cfg.save_animation
                    )
                )
                # Export to file.
                if test_cfg.export_to_file:
                    export_rewards_2_file(model_id=model_id, rewards=rewards[-1], reward_per_episode=test_cfg.reward_per_episode)
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
                            sequence_length=test_cfg.sequence_length,
                            reward_per_episode=test_cfg.reward_per_episode,
                            export_animation=test_cfg.save_animation
                        )
                    )
                    # Export to file.
                    if test_cfg.export_to_file:
                        export_rewards_2_file(model_id=model_id, rewards=rewards[-1], reward_per_episode=test_cfg.reward_per_episode)
                        export_config_2_json_file(config=test_cfg, file_name=f'test_config-{model_id}', path=os.path.join(models_path, model_id))
            else:
                warnings.warn(f'No models found in {models_path}.')

    else:
        pass

    return rewards

def test_model_on_multiple_envs(model_id:str, nb_test_episodes:int, sequence_length:int, reward_per_episode:str, export_animation:bool=False) -> dict:
    """
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Instantiate model
    # -------------------------------------------
    model = load_model(model_id=model_id)
    device = get_model_device(model_id=model_id)

    # Loading training dataset (prevent data leakage)
    # -------------------------------------------
    dataset_name = get_model_dataset(model_id=model_id) # dataset (.pt) file used for training.
    dataset = torch.load(os.path.join(project_root, 'data', 'processed', dataset_name), weights_only=False)
    
    # Simulate env
    # -------------------------------------------
    test_iter = range(dataset.nb_episodes, dataset.nb_episodes+nb_test_episodes)
    progress = tqdm(test_iter, desc=f'Testing model ID {model_id}', unit='env(seed)')

    rewards_per_env_seed = {}
    for rand_seed in progress:

        # play env
        frames, reward_per_step, _, _, _ = play_env(device, rand_seed, model, dataset, "rgb_array" if export_animation else None, sequence_length)
        rewards_per_env_seed[f'env_seed_{rand_seed}'] = np.sum(reward_per_step) if reward_per_episode == 'total' else np.array(reward_per_step)

        progress.set_postfix({'accumulated reward': f'{np.sum(reward_per_step):.3f}'})

        if export_animation:
            save_animation(
                frames=frames, modeL_dir=os.path.join(project_root, 'results', 'models', model_id), file_name=f'env_seed_{rand_seed}_{int(np.sum(reward_per_step))}.gif'
            )

    return rewards_per_env_seed

def play_env(device, rand_seed, model, dataset, mode, sequence_length):
    env = gym.make("LunarLander-v3", render_mode=mode)
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

def export_rewards_2_file(model_id:str, rewards:dict, reward_per_episode:str) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    models_path = os.path.join(project_root, 'results', 'models', model_id)

    with open(os.path.join(models_path, f'evaluation-{model_id}-reward_per_episode_{reward_per_episode}.pkl'), 'wb') as file:
        pickle.dump(rewards, file)

def test_hpo(test_cfg):
    from src.hpo.custom_fitness_functions import evaluate_lander_performance

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Instantiate model
    # -------------------------------------------
    model = load_model(model_id=test_cfg.model_id)
    device = get_model_device(model_id=test_cfg.model_id)

    # Loading training dataset (prevent data leakage)
    # -------------------------------------------
    dataset_name = get_model_dataset(model_id=test_cfg.model_id) # dataset (.pt) file used for training.
    dataset = torch.load(os.path.join(project_root, 'data', 'processed', dataset_name), weights_only=False)
    
    # Simulate env
    # -------------------------------------------
    test_iter = range(dataset.nb_episodes, dataset.nb_episodes+test_cfg.nb_test_episodes)
    progress = tqdm(test_iter, desc=f'Testing model ID {test_cfg.model_id}', unit='env(seed)')

    fitnesses = []
    for rand_seed in progress:

        # play env
        frames, reward_per_step, states, _, _ = play_env(device, rand_seed, model, dataset, None, test_cfg.sequence_length)
        
        fitnesses.append(evaluate_lander_performance(states[-1], reward_per_step))

        progress.set_postfix({'accumulated reward': f'{np.sum(reward_per_step):.3f}'})

    return fitnesses