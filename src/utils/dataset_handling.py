import torch
from torch.utils.data import Subset

def subset_dataset(dataset, nb_samples:int, seed:int=0) -> Subset:
    """
    Returns a new dataset containing exactly `nb_samples` elements randomly
    selected from the original dataset.

    Args:
        dataset (Dataset): Any instance of torch.utils.data.Dataset.
        nb_samples (int): Number of samples to include in the subset.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Subset: A torch.utils.data.Subset object containing nb_samples samples.
    """
    if nb_samples > len(dataset):
        raise ValueError(f"Requested {nb_samples} samples, but dataset has only {len(dataset)}.")

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:nb_samples].tolist()
    return Subset(dataset, indices)
