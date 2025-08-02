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

import os
from glob import glob


import monai
from monai.data import list_data_collate, DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)


def create_tumor_dataset(
    dataset_dir: str = "data/raw/tumor-segmentation",
    val_size: float = 0.8,
    train_transforms: Compose = None,
    val_transforms: Compose = None,
) -> tuple[monai.data.Dataset, monai.data.Dataset]:
    """
    Create a dataset for tumor segmentation.

    Args:
        dataset_dir (str): Path to the dataset directory.
        val_size (float): Proportion of the dataset to include in the validation split (default is 0.8).
        train_transforms (Compose): Transforms to apply to the training dataset.
        val_transforms (Compose): Transforms to apply to the validation dataset.
    Returns:
        tuple[monai.data.Dataset, monai.data.Dataset]: Training and validation datasets.
    """
    # Configure directories
    controls_imgs = sorted(glob(os.path.join(dataset_dir, "controls", "imgs", "*.png")))
    print(f"Found {len(controls_imgs)} control images.")
    patient_imgs = sorted(glob(os.path.join(dataset_dir, "patients", "imgs", "*.png")))
    patient_segs = sorted(
        glob(os.path.join(dataset_dir, "patients", "labels", "*.png"))
    )

    # 2) Lag liste over filpar for trening og validering
    #    Vi bruker kun pasientene her (fordi controls mangler ekte mask)
    all_pairs = [
        {"img": img, "seg": seg} for img, seg in zip(patient_imgs, patient_segs)
    ]

    # train_test_split(all_pairs, test_size=val_size, random_state=42)
    mid = len(all_pairs) // 2
    train_files = all_pairs[:mid]
    val_files = all_pairs[mid:]

    default_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )
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
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        ]
    )
    val_transforms = Compose([*default_transforms])

    # 4) Datasets og dataloaders
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

    return train_ds, val_ds


def create_tumor_dataloader(
    dataset: monai.data.Dataset,
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
