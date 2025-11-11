import re
import numpy as np

def get_top_n_rewards(eval_data:dict, n:int=3) -> dict:
    """ Gets the best rewards of a model.

    Reads through the evaluation data of a model (i.e., model has been tested/deployed on the environment
    the expert controls and the model learns from) and returns the `n` best rewards achieved.

    Args:
        eval_data (dict): A dictionary where `key` gives the seed used for the evaluation env. (in the
            form `env_seed_<seed>`) and `value` is either the reward per env. step (np.array) or the 
            accumulated reward for the entire episode (single value).
        n (int): The top `n` rewards to be returned.
    Returns:
        top_n_data (dict): A dictionary where the `key` is an integer giving the seed used for the env.
            and `value` is the accumulated reward for the episode.
    Raises:
        ValueError: The number `n` of top rewards exceeds the number of episodes the model was evaluated on.
    """
    data = dict()

    if n > len(eval_data.keys()):
        raise ValueError(f'Top n rewards set to {n} but model tested on only {len(eval_data.keys())} environment seeds.')

    # Read through reward data per environment tested
    for env_info, env_reward_data in eval_data.items():
        env_seed = int(re.findall(r'\d+', env_info)[0])

        if isinstance(env_reward_data, np.ndarray):
            accumulated_reward = int(np.sum(env_reward_data))
        elif isinstance(env_reward_data, (float, int, np.number)):
            accumulated_reward = int(env_reward_data)
        else:
            raise TypeError(f'Unsupported type for rewards: {type(env_reward_data)}.')
        
        data[env_seed] = accumulated_reward

    # Sort based on accumulated reward
    data = dict(
        sorted(data.items(), key=lambda item: item[1], reverse=True)
    )

    top_n_data = dict(list(data.items())[:n])

    return top_n_data