import os
import os
import shutil

def save_animation(frames:list, modeL_dir:os.path, file_name:str, overrride_existing:bool=True) -> None:
    import imageio
    
    gif_path = os.path.join(modeL_dir, 'animations')

    # Override existing animations
    if overrride_existing and os.path.exists(gif_path) and os.path.isdir(gif_path):
        shutil.rmtree(gif_path)

    os.makedirs(gif_path, exist_ok=True)
    imageio.mimsave(os.path.join(gif_path, file_name), frames, fps=30)