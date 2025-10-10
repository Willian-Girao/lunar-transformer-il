# -*- coding: utf-8 -*-
def main():
    import gymnasium as gym
    import torch, pickle, os
    import numpy as np
    import argparse
    from src.models.lander_actor_critic.ActorCritic import ActorCritic
    import warnings
    warnings.filterwarnings("ignore")
    np.bool = np.bool_

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--max_steps', type=int, help='Maximum number of steps an episode can last.', default=400)
    parser.add_argument('--nb_episodes', type=int, help='Number of episodes.', default=1000)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Configuration
    # -------------------------------------------
    max_steps = args.max_steps
    nb_episodes = args.nb_episodes

    # Instantiate Actor-Critic model
    # -------------------------------------------
    policy = ActorCritic()
    policy.load_state_dict(torch.load(
        os.path.join(project_root, 'src', 'models', 'lander_actor_critic', 'actorcritic_lunarlander', 'LunarLander_TWO.pth'))
    )

    export_path = os.path.join(project_root, 'data', 'raw')

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    lunarlander_training_data = {'X': [], 'padding_idxs': [], 'Y': [], 'rewards': []}

    # Multiple envs with different initial states
    # -------------------------------------------
    seed = 0
    while len(lunarlander_training_data['X']) < nb_episodes:
        steps = 0
        env = gym.make("LunarLander-v3", render_mode=None)

        np.random.seed(seed)
        observation, info = env.reset(seed=seed)

        states = []
        actions = []
        episode_reward = 0.0

        # Single environment simulation
        # -----------------------------
        done = False
        while not done:
            steps += 1
            # Take action and update state space
            # ----------------------------------
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            actions.append(action)
            states.append(observation)
            episode_reward += reward

            done = terminated or truncated

            if terminated and not truncated:
                if len(states) <= max_steps:
                    # marks where padding starts for the dataset.
                    padding_idx = len(states)

                    # make sure sequences have all the same length.
                    while len(states) < max_steps:
                        states.append(states[-1])
                        actions.append(actions[-1])
                    
                    lunarlander_training_data['X'].append(np.array(states))
                    lunarlander_training_data['padding_idxs'].append(padding_idx)
                    lunarlander_training_data['Y'].append(np.array(actions))
                    lunarlander_training_data['rewards'].append(episode_reward)

        env.close()
        seed += 1

        percent = int((len(lunarlander_training_data['X'])/nb_episodes)*100)
        print(f'Episodes: {percent:03}%', end='\r', flush=True)

    # Save data to file
    # -------------------------------------------
    lunarlander_training_data['X'] = np.array(lunarlander_training_data['X'])
    lunarlander_training_data['padding_idxs'] = np.array(lunarlander_training_data['padding_idxs'])
    lunarlander_training_data['Y'] = np.array(lunarlander_training_data['Y'])
    lunarlander_training_data['rewards'] = np.array(lunarlander_training_data['rewards'])

    with open(os.path.join(export_path, f'gymnasium-ActorCritic-LunarLander-{nb_episodes}.pkl'), 'wb') as file:
        pickle.dump(lunarlander_training_data, file)

if __name__ == "__main__":
    main()