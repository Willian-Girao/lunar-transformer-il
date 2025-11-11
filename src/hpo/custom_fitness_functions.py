import numpy as np

def evaluate_lander_performance(last_state, rewards_per_step, w_slope=1.0, w_landing=1.0, w_reward=0.5):
    """
    Compute a fitness metric for Lunar Lander based on:
    - upward trend of accumulated reward curve
    - successful landing (both legs touching)
    - accumulated reward magnitude
    
    Args:
        last_state (np.ndarray): Final environment state (size 8 for LunarLander-v3)
                                 last_state[-2:] correspond to left/right leg contact (1 or 0)
        rewards_per_step (list or np.ndarray): Reward at each step
        w_slope, w_landing, w_reward: weights for each component
    
    Returns:
        float: Fitness score (to be maximized)
    """

    rewards = np.array(rewards_per_step, dtype=float)
    accumulated = np.cumsum(rewards)

    # ----- 1. Trend: slope of accumulated reward curve -----
    x = np.arange(len(accumulated))
    if len(x) > 1:
        slope, _ = np.polyfit(x, accumulated, 1)
    else:
        slope = 0.0

    # Normalize slope to a reasonable range
    slope_score = np.tanh(slope / 10.0)

    # ----- 2. Landing success -----
    landing_success = 1.0 if (last_state[-2] == 1 and last_state[-1] == 1) else 0.0

    # ----- 3. Total accumulated reward -----
    total_reward = accumulated[-1] if len(accumulated) > 0 else 0.0
    # Scale to [-1, 1] range for stability
    reward_score = np.tanh(total_reward / 200.0)

    # Combine with weights
    fitness = (
        w_slope * slope_score +
        w_landing * landing_success +
        w_reward * reward_score
    )

    # Penalize if lander did not land and total reward is strongly negative
    if landing_success == 0 and total_reward < 0:
        fitness *= 0.5  # dampen bad crashes

    return fitness

def evaluate_fitness_with_AUC(last_state, rewards_per_step, norm_scale=500.0):
    """
    Compute a simplified fitness score for the Lunar Lander task based on:
    - Area under the reward curve (overall cumulative performance)
    - Successful landing (both legs touching)

    Args:
        last_state (np.ndarray): Final environment state (size 8 for LunarLander-v3)
                                 last_state[-2:] correspond to left/right leg contact (1 or 0)
        rewards_per_step (list or np.ndarray): Reward at each step
        w_area (float): Weight for the area under reward curve component
        w_landing (float): Weight for landing success component
        norm_scale (float): Used to normalize area (based on maximum negative reward observed).

    Returns:
        float: Fitness score (to be maximized)
    """
    rewards = np.array(rewards_per_step, dtype=float)

    # ----- 1. Area under reward curve -----
    # Equivalent to total accumulated reward, but integrates step-by-step
    if len(rewards) > 0:
        area = np.trapz(rewards)  # trapezoidal numerical integration
    else:
        area = 0.0

    # ----- 2. Landing success -----
    landing_success = True if (last_state[-2] == 1 and last_state[-1] == 1) else False
    if not landing_success:
        area *= 0.5

    # Normalize area to [-1, 1]
    area_score = np.tanh(area / norm_scale)

    return area_score
