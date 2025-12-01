# -*- coding: utf-8 -*-
def main():
    import gymnasium as gym
    import torch, os
    import numpy as np
    import argparse
    from src.models.lander_actor_critic.ActorCritic import ActorCritic
    from src.evaluation.TestingConfig import TestingConfig
    from src.utils.gym_env_handling import sample_env_setting
    import warnings
    from tqdm import tqdm
    from src.utils.gym_env_handling import save_animation
    warnings.filterwarnings("ignore")
    np.bool = np.bool_

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_json', type=str, help='Configuration .json file describing testing hyperparameters.')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Instantiate Actor-Critic model
    # -------------------------------------------
    policy = ActorCritic()
    policy.load_state_dict(torch.load(
        os.path.join(project_root, 'src', 'models', 'lander_actor_critic', 'actorcritic_lunarlander', 'LunarLander_TWO.pth'))
    )

    # Load the test loop configuration file
    # -------------------------------------------
    test_cfg = TestingConfig()
    test_cfg.from_json(json_file=os.path.join(project_root, 'configs', args.test_json))

    test_iter = range(1000, 1000+test_cfg.nb_test_episodes) # TODO: hacky - I'm assuming that the first 1k seeds are used to generate training data. Implement this better.
    progress = tqdm(test_iter, desc=f'Testing expert', unit='env(seed)')

    dir = f'env_setup-coef_of_var_{test_cfg.env_coef_of_var}' + f'_seed_{test_cfg.seed_coef_of_var}' if test_cfg.env_coef_of_var != 0 else ''
    gif_path = os.path.join(project_root, 'results', 'expert', dir)

    # Multiple environment simulation
    # -----------------------------
    for seed in progress:

        if test_cfg.env_coef_of_var == 0:
            env = gym.make("LunarLander-v3", render_mode="rgb_array")
        else:
            (gravity, wind_power, turbulence_power) = sample_env_setting(
                coef_var=test_cfg.env_coef_of_var,
                seed=test_cfg.seed_coef_of_var
            )

            env = gym.make(
                "LunarLander-v3",
                render_mode="rgb_array",
                enable_wind=True,
                gravity=gravity,
                wind_power=wind_power,
                turbulence_power=turbulence_power
            )

        np.random.seed(seed)
        observation, info = env.reset(seed=seed)
        episode_reward = 0.0

        frames = []
        done = False
        while not done:
            # Take action and update state space
            # ----------------------------------
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward

            done = terminated or truncated

            frame = env.render()
            frames.append(frame)

        env.close()

        progress.set_postfix({'episode total reward': f'{int(episode_reward)}'})

        save_animation(
            frames=frames, path=gif_path, file_name=f'env_seed_{seed}_{int(episode_reward)}.gif'
        )

if __name__ == "__main__":
    main()