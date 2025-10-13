import os

def save_animation(frames:list, modeL_dir:os.path, file_name:str) -> None:
    import imageio
    
    gif_path = os.path.join(modeL_dir, 'animations')
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimsave(os.path.join(gif_path, file_name), frames, fps=30)