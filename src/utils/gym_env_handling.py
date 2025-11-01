import os

def save_animation(frames:list, path:os.path, file_name:str) -> None:
    import imageio

    os.makedirs(path, exist_ok=True)
    imageio.mimsave(os.path.join(path, file_name), frames, fps=30)