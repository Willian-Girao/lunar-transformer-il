# -*- coding: utf-8 -*-
def main():
    import gymnasium as gym
    import torch, os
    import numpy as np
    import argparse
    from src.models.lander_actor_critic.ActorCritic import ActorCritic
    import warnings
    from src.utils.gym_env_handling import save_animation
    warnings.filterwarnings("ignore")
    np.bool = np.bool_

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, help='', default=0)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Instantiate Actor-Critic model
    # -------------------------------------------
    policy = ActorCritic()
    policy.load_state_dict(torch.load(
        os.path.join(project_root, 'src', 'models', 'lander_actor_critic', 'actorcritic_lunarlander', 'LunarLander_TWO.pth'))
    )

    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    np.random.seed(args.seed)
    observation, info = env.reset(seed=args.seed)
    episode_reward = 0.0

    # Single environment simulation
    # -----------------------------
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

    gif_path = os.path.join(project_root, 'results', 'expert', f'expert-env_seed_{args.seed}')

    save_animation(
        frames=frames, path=gif_path, file_name=f'env_seed_{args.seed}_{int(episode_reward)}.gif'
    )

if __name__ == "__main__":
    main()