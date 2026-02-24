# -*- coding: utf-8 -*-
def main():
    import gymnasium as gym
    import torch, pickle, os
    import numpy as np
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    np.bool = np.bool_

    from src.models.cheetah_sac.sac import SACAgent

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--max_steps', type=int, help='Maximum number of steps an episode can last.', default=1000)
    parser.add_argument('--nb_episodes', type=int, help='Number of episodes.', default=1000)
    parser.add_argument('--start_env_seed', type=int, help='', default=0)
    parser.add_argument("--is_eval", action="store_true", help='...')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['human', 'none'],
        default=None,
        help='Model type: expert or transformer.'
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Configuration
    # -------------------------------------------
    max_steps = args.max_steps
    nb_episodes = args.nb_episodes

    export_path = os.path.join(project_root, 'data', 'raw')

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    halfcheetah_training_data = {'X': [], 'padding_idxs': [], 'Y': [], 'rewards': []}

    # Instantiate expert model and env
    # -------------------------------------------
    envName = "HalfCheetah-v5"
    env = gym.make(envName, render_mode="human" if args.mode != None else None)

    action_space = env.action_space.shape[0]
    state_space = env.observation_space.shape[0] 
    
    agent = SACAgent(state_space, action_space, reward_scale=2, tau=0.005)
    agent.load_models()

    env.close()

    # Multiple envs with different initial states
    # -------------------------------------------
    seed = args.start_env_seed
    while len(halfcheetah_training_data['X']) < nb_episodes:
        env = gym.make(envName, render_mode="human" if args.mode != None else None)

        np.random.seed(seed)
        observation, info = env.reset(seed=seed)

        states = []
        actions = []
        episode_reward = 0.0

        # Running environment
        # -------------------------------------------
        state, _ = env.reset()
        r = 0
        step = 0

        obs, _ = env.reset()
        done = False
        while (not done) or (step <= args.max_steps):
            action = agent.choose_action(obs.tolist())
            obs, reward, terminated, truncated, _ = env.step(action)

            actions.append(action)
            states.append(obs)
            episode_reward += reward

            done = terminated or truncated

            if (terminated and not truncated) or (step == args.max_steps-1):
                done = True
                if len(states) <= max_steps:
                    # marks where padding starts for the dataset.
                    padding_idx = len(states)

                    # make sure sequences have all the same length.
                    while len(states) < max_steps:
                        states.append(states[-1])
                        actions.append(actions[-1])
                    
                    halfcheetah_training_data['X'].append(np.array(states))
                    halfcheetah_training_data['padding_idxs'].append(padding_idx)
                    halfcheetah_training_data['Y'].append(np.array(actions))
                    halfcheetah_training_data['rewards'].append(episode_reward)

            if args.mode != None:
                env.render()
            step += 1

        env.close()
        seed += 1

        percent = int((len(halfcheetah_training_data['X'])/nb_episodes)*100)
        print(f'Episodes: {percent:03}% (reward: {int(episode_reward)} | episode: {len(halfcheetah_training_data['X'])})', end='\r', flush=True)

    # Save data to file
    # -------------------------------------------
    halfcheetah_training_data['X'] = np.array(halfcheetah_training_data['X'])
    halfcheetah_training_data['padding_idxs'] = np.array(halfcheetah_training_data['padding_idxs'])
    halfcheetah_training_data['Y'] = np.array(halfcheetah_training_data['Y'])
    halfcheetah_training_data['rewards'] = np.array(halfcheetah_training_data['rewards'])

    with open(os.path.join(export_path, f'{'evaluation-' if args.is_eval else ''}gymnasium-SAC-HalfCheetah-{nb_episodes}.pkl'), 'wb') as file:
        pickle.dump(halfcheetah_training_data, file)

if __name__ == "__main__":
    main()