# -*- coding: utf-8 -*-
def main():
    import gymnasium as gym
    import torch, os, pickle
    import numpy as np
    import argparse
    from src.models.lander_actor_critic.ActorCritic import ActorCritic
    from src.evaluation.TestingConfig import TestingConfig
    from src.utils.gym_env_handling import sample_env_setting
    from src.utils.evaluation_handling import export_rewards_2_file
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

    test_iter = range(test_cfg.nb_seeds)
    progress = tqdm(test_iter, desc=f'across seed variation (expert)', unit='')

    # File export config
    # -------------------------------------------
    export_path = os.path.join(project_root, 'results', 'expert', 'within_seed_var')
    os.makedirs(export_path, exist_ok=True)

    env_setup = f'env_setup-coef_of_var_{test_cfg.env_coef_of_var}'
    env_setup += f'_seed_{test_cfg.seed_coef_of_var}' if test_cfg.env_coef_of_var != 0 else ''

    # Across seeds
    # -----------------------------
    test_cfg.env_seed = test_cfg.starting_seed
    for s in progress:

        within_iter = range(test_cfg.nb_episodes)
        within_seed_progress = tqdm(within_iter, desc=f'within seed variation (seed {test_cfg.env_seed})', unit='', leave=False)

        # Within seeds
        # -----------------------------
        R_s_r = {}
        for r in within_seed_progress:

            if test_cfg.env_coef_of_var == 0:
                env = gym.make(
                    "LunarLander-v3",
                    render_mode=None
                )
            else:
                (gravity, wind_power, turbulence_power) = sample_env_setting(
                    coef_var=test_cfg.env_coef_of_var,
                    seed=test_cfg.seed_coef_of_var
                )

                env = gym.make(
                    "LunarLander-v3",
                    render_mode=None,
                    enable_wind=True,
                    gravity=gravity,
                    wind_power=wind_power,
                    turbulence_power=turbulence_power
                )

            np.random.seed(test_cfg.env_seed)
            observation, info = env.reset(seed=test_cfg.env_seed)
            
            reward_per_step = []
            done = False
            while not done:
                # Take action and update state space
                # ----------------------------------
                action = policy(observation)
                observation, reward, terminated, truncated, info = env.step(action)

                reward_per_step.append(reward)

                done = terminated or truncated

            R_s_r[(test_cfg.env_seed, r)] = np.sum(reward_per_step) if test_cfg.reward_per_episode == 'accumulated' else np.array(reward_per_step)

            env.close()

        # Export within seed data
        # ----------------------------------
        with open(
            os.path.join(
                export_path,
                f'env_seed_{test_cfg.env_seed}-reward_per_episode_{test_cfg.reward_per_episode}{env_setup}.pkl'
                ), 'wb'
            ) as file:
            pickle.dump(R_s_r, file)

        test_cfg.env_seed += 1

if __name__ == "__main__":
    main()