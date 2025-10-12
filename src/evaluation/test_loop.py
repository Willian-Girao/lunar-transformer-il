from src.utils.model_handling import load_model, get_model_dataset, update_state_action_buffers, get_model_device, init_state_action_buffers
import torch, os
import gymnasium as gym
import numpy as np

def test(test_cfg):
    """
    """
    rewards = None

    if isinstance(test_cfg.model, str):                 # id of a single model to be tested.
        rewards = test_single_model(
            model_id=test_cfg.model,
            nb_test_episodes=test_cfg.nb_test_episodes,
            sequence_length=test_cfg.sequence_length,
            reward_per_episode=test_cfg.reward_per_episode,
            export_animation=test_cfg.save_animation
        )

        print(rewards)
    elif isinstance(test_cfg.model, list):
        if len(test_cfg.model):
            # no model ids in the list - test all in `/results/models`.
            pass
        else:
            # list of one or more model ids to be tested.
            pass
    else:
        pass

    return rewards

def test_single_model(model_id:str, nb_test_episodes:int, sequence_length:int, reward_per_episode:str, export_animation:bool=False) -> np.array:
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
    print(dataset_name)
    dataset = torch.load(os.path.join(project_root, 'data', 'processed', dataset_name), weights_only=False)
    
    # Simulate env
    # -------------------------------------------
    rewards = []
    for rand_seed in range(dataset.nb_episodes, dataset.nb_episodes+nb_test_episodes):
        reward_per_step = []

        env = gym.make("LunarLander-v3", render_mode="rgb_array" if export_animation else None)
        state, info = env.reset(seed=rand_seed)
        done = False

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

        env.close()
        rewards.append(np.sum(reward_per_step) if reward_per_episode == 'total' else np.array(reward_per_step))

        if export_animation:
            save_animation(
                frames=frames, modeL_dir=os.path.join(project_root, 'results', 'models', model_id), file_name=f'env_seed_{rand_seed}.gif'
            )

    return np.array(rewards)

def save_animation(frames:list, modeL_dir:os.path, file_name:str) -> None:
    import imageio
    
    gif_path = os.path.join(modeL_dir, 'animations')
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimsave(os.path.join(gif_path, file_name), frames, fps=30)