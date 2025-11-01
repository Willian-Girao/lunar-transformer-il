import re
import numpy as np

def get_top_n_rewards(eval_data:dict, n:int=3) -> dict:
    """
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