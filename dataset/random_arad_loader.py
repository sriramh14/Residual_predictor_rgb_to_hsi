import os
import random
from torch.utils.data import Subset

from dataset.dataset_loader import ARADDataset


def load_random_arad1k_samples(
    root_dir="data",
    num_samples=50,
    seed=42,
    total_images=1000,
    cube_key="cube",
    download=True
):
    """
    Build an ARAD1K RGB-HSI dataset and return a reproducible random subset.

    Parameters
    ----------
    root_dir : str
        Directory containing (or receiving) the ARAD1K files.
    num_samples : int
        Number of unique RGB-HSI pairs to select.
    seed : int
        Random seed used to select the samples.
    total_images : int
        Maximum number of ARAD1K pairs requested from the base loader.
    cube_key : str
        MATLAB key containing the hyperspectral cube.
    download : bool
        Download missing files through the base ARADDataset loader.

    Returns
    -------
    subset : torch.utils.data.Subset
        Dataset containing only the selected random samples.
    selected_samples : list[dict]
        Metadata containing subset position, original dataset index,
        RGB filename, and HSI filename.
    """

    # train=True with train_images=total_images exposes the complete pool
    # prepared by ARADDataset instead of applying its train/validation split.
    full_dataset = ARADDataset(
        root_dir=root_dir,
        train=True,
        train_images=total_images,
        total_images=total_images,
        cube_key=cube_key,
        download=download
    )

    available_samples = len(full_dataset)

    if available_samples == 0:
        raise RuntimeError(
            "No paired ARAD1K RGB-HSI samples were found. "
            "Check root_dir and the dataset directory structure."
        )

    if num_samples > available_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but only "
            f"{available_samples} paired samples are available."
        )

    rng = random.Random(seed)
    selected_indices = rng.sample(
        range(available_samples),
        num_samples
    )

    subset = Subset(
        full_dataset,
        selected_indices
    )

    selected_samples = []

    for subset_index, dataset_index in enumerate(selected_indices):
        hsi_path, rgb_path = full_dataset.pairs[dataset_index]

        selected_samples.append(
            {
                "subset_index": subset_index,
                "dataset_index": dataset_index,
                "rgb_filename": os.path.basename(rgb_path),
                "hsi_filename": os.path.basename(hsi_path)
            }
        )

    print(
        f"Randomly selected {num_samples} samples "
        f"from {available_samples} available ARAD1K pairs "
        f"using seed {seed}"
    )

    return subset, selected_samples
