import torch
from torch.utils.data import Subset, Dataset

def subset_dataset(dataset:Dataset, nb_samples:int, seed:int=0) -> Subset:
    """ Subsets a `torch.utils.Dataset` instance.
    
    Returns a new dataset containing exactly `nb_samples` elements randomly
    selected from the original dataset.

    Args:
        dataset (Dataset): Any instance of torch.utils.data.Dataset.
        nb_samples (int): Number of samples to include in the subset.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Subset: A torch.utils.data.Subset object containing nb_samples samples.

    Raises:
        ValueError: The number of samples in the subset is higher than the samples in the dataset.
    """
    if nb_samples > len(dataset):
        raise ValueError(f"Requested {nb_samples} samples, but dataset has only {len(dataset)}.")

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:nb_samples].tolist()
    return Subset(dataset, indices)