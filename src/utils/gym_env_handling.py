import os

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