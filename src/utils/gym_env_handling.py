import os
import numpy as np

def save_animation(frames:list, path:os.path, file_name:str) -> None:
    """ Exports an env. (episode) to file.

    A folder specified by `path` is created and the animation file is exported
    with the name `file_name`.

    Args:
        frames (list): A list of frames with each step of an environment simulation.
        path (os.path): The path (usually whithin a model`s directory) where the animation will be saved.
        file_name (str): The name of the animation (ending with a `.gif`).
    """
    import imageio

    os.makedirs(path, exist_ok=True)
    imageio.mimsave(os.path.join(path, file_name), frames, fps=30, loop=0)

def sample_env_setting(coef_var:float, seed:int):
    """Samples values for env. gravity, wind_power, and turbulence_power.

    Because these parameters (gravity, wind, turbulence) have different scales, we use the coeficient
    of variation as an argument since it naturally scales the noise to the magnitude of the parameter.

    Args:
        coef_var (float): the coefficient of variation (in percentage) used to
            sample new values from their defaults. E.g., coef_var=10 means
            10% relative standard deviation.
        seed (int): random seed for reproducible sampling.

    Returns:
        tuple: (new_gravity, new_wind_power, new_turbulence_power)

    Raises:
        ValueError: if `coef_var` is not between 0 and 100.
    """

    if not (0 <= coef_var <= 100):
        raise ValueError("coef_var must be between 0 and 100 (percentage).")
    
    # Create independent RNG instance
    rng = np.random.default_rng(seed)

    # Default values used in the default env instantiation.
    gravity = -10.0
    wind_power = 15.0
    turbulence_power = 1.5

    # Convert coefficient of variation (%) to a multiplier
    cv = coef_var / 100.0

    def sample(mu):
        """Sample from a normal distribution with std = CV * |mu|."""
        sigma = cv * abs(mu)
        return rng.normal(mu, sigma)

    new_gravity = -abs(sample(gravity))
    new_wind_power = max(0.0, sample(wind_power))
    new_turbulence_power = max(0.0, sample(turbulence_power))

    return new_gravity, new_wind_power, new_turbulence_power