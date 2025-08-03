"""
Utility Dataset and Dataloader builders for a tumor segmentation task.

Directory structure expected (relative to the project root)::

    data/
        raw/
            tumor-segmentation/
                controls/
                    imgs/*.png
                patients/
                    imgs/*.png
                    labels/*.png

Controls ship without ground-truth masks; if you choose to include them they will automatically receive an all-zero mask so that the network can be trained with explicit negatives.
"""

import math
import os
import random
from glob import glob

from monai.data import list_data_collate, DataLoader, Dataset
from monai.transforms import (
    EnsureChannelFirstd,
    Transform,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
)


def create_tumor_dataset(
    dataset_dir: str = "data/raw/tumor-segmentation",
    val_size: float = 0.8,
    train_data_augmentation: list[Transform] = [],
    val_data_augmentations: list[Transform] = [],
) -> tuple[Dataset, Dataset]:
    """
    Create a dataset for tumor segmentation.

    Args:
        dataset_dir (str): Path to the dataset directory.
        val_size (float): Proportion of the dataset to include in the validation split from 0 to 1 (default is 0.8).
        train_transforms (Compose): Transforms to apply to the training dataset.
        val_transforms (Compose): Transforms to apply to the validation dataset.
    Returns:
        tuple[monai.data.Dataset, monai.data.Dataset]: Training and validation datasets.
    """
    # Configure directories
    controls_imgs = sorted(glob(os.path.join(dataset_dir, "controls", "imgs", "*.png")))
    control_masks = sorted(
        glob(os.path.join(dataset_dir, "controls", "labels", "*.png"))
    )
    print(f"Found {len(controls_imgs)} control images.")
    patient_imgs = sorted(glob(os.path.join(dataset_dir, "patients", "imgs", "*.png")))
    patient_segs = sorted(
        glob(os.path.join(dataset_dir, "patients", "labels", "*.png"))
    )
    print(f"Found {len(patient_imgs)} patient images.")

    # Create pairs for both patients and controls
    patient_pairs = [
        {"img": img, "seg": seg} for img, seg in zip(patient_imgs, patient_segs)
    ]

    control_pairs = [
        {"img": img, "seg": seg} for img, seg in zip(controls_imgs, control_masks)
    ]

    # Use all patient data and randomly sample equal number of controls for 50/50 split
    num_patients = len(patient_pairs)

    if len(control_pairs) >= num_patients:
        # Randomly sample control pairs to match the number of patient pairs
        random.seed(42)  # For reproducibility
        selected_control_pairs = random.sample(control_pairs, num_patients)
        print(
            f"Randomly selected {num_patients} control samples from {len(control_pairs)} available."
        )
    else:
        # Use all available control pairs if there are fewer than patient pairs
        selected_control_pairs = control_pairs
        print(
            f"Warning: Only {len(control_pairs)} control samples available, less than {num_patients} patient samples."
        )

    # Concatenate all data (patients + selected controls)
    all_pairs = patient_pairs + selected_control_pairs
    print(
        f"Final dataset: {len(patient_pairs)} patients + {len(selected_control_pairs)} controls = {len(all_pairs)} samples (50/50 split)"
    )

    split = math.floor(len(all_pairs) * val_size)
    train_files = all_pairs[:split]
    val_files = all_pairs[split:]

    default_transforms = [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
    ]
    # 3) Transforms
    train_transforms = Compose(
        [
            *default_transforms,
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=[96, 96],
                pos=1,
                neg=1,
                num_samples=4,
            ),
            *train_data_augmentation,
        ]
    )
    val_transforms = Compose([*default_transforms, *val_data_augmentations])

    # 4) Datasets og dataloaders
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    return train_ds, val_ds


def create_tumor_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the tumor segmentation dataset.

    Args:
        dataset (monai.data.Dataset): The dataset to create a DataLoader for.
        batch_size (int): Number of samples per batch (default is 4).
        num_workers (int): Number of subprocesses to use for data loading (default is 0).
        shuffle (bool): Whether to shuffle the data at every epoch (default is True).
    Returns:
        DataLoader: A DataLoader for the tumor segmentation dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        shuffle=shuffle,
    )
