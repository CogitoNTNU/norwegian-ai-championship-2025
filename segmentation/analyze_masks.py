#!/usr/bin/env python3
"""
Quick analysis of generated masks to check if they're truly black
"""

import cv2
import numpy as np
import os
from glob import glob


def analyze_mask_files(dataset_dir: str = "../data/raw/tumor-segmentation"):
    """Analyze the generated control masks to see what's actually in them"""

    # Check if directory exists
    if not os.path.exists(dataset_dir):
        dataset_dir = "data/raw/tumor-segmentation"

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    controls_labels_dir = os.path.join(dataset_dir, "controls", "labels")

    if not os.path.exists(controls_labels_dir):
        print(f"Controls labels directory not found: {controls_labels_dir}")
        return

    # Get mask files
    mask_files = sorted(glob(os.path.join(controls_labels_dir, "*.png")))

    if not mask_files:
        print("No mask files found!")
        return

    print(f"Found {len(mask_files)} mask files")
    print("Analyzing first few masks...")

    for i, mask_path in enumerate(mask_files[:3]):  # Analyze first 3 masks
        print(f"\n--- Mask {i + 1}: {os.path.basename(mask_path)} ---")

        # Load with different flags to see the difference
        mask_unchanged = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        print(
            f"IMREAD_UNCHANGED: shape={mask_unchanged.shape if mask_unchanged is not None else 'None'}, dtype={mask_unchanged.dtype if mask_unchanged is not None else 'None'}"
        )
        print(
            f"IMREAD_GRAYSCALE: shape={mask_grayscale.shape if mask_grayscale is not None else 'None'}, dtype={mask_grayscale.dtype if mask_grayscale is not None else 'None'}"
        )
        print(
            f"IMREAD_COLOR: shape={mask_color.shape if mask_color is not None else 'None'}, dtype={mask_color.dtype if mask_color is not None else 'None'}"
        )

        if mask_unchanged is not None:
            # Check unique values
            unique_vals = np.unique(mask_unchanged)
            print(f"Unique values: {unique_vals}")

            # Check if it's all zeros (black)
            if np.all(mask_unchanged == 0):
                print("✓ Mask is completely black (all zeros)")
            else:
                print("⚠ Mask is NOT completely black!")

            # Check for alpha channel
            if len(mask_unchanged.shape) == 3:
                if mask_unchanged.shape[2] == 4:
                    print("Has alpha channel (RGBA)")
                    print(
                        f"Alpha channel unique values: {np.unique(mask_unchanged[:, :, 3])}"
                    )
                elif mask_unchanged.shape[2] == 3:
                    print("RGB format (no alpha)")
                else:
                    print(f"Unexpected number of channels: {mask_unchanged.shape[2]}")
            else:
                print("Grayscale format")

            # Check min/max values
            print(f"Min value: {np.min(mask_unchanged)}")
            print(f"Max value: {np.max(mask_unchanged)}")

    # Also check a patient mask for comparison
    print(f"\n{'=' * 50}")
    print("COMPARING WITH PATIENT MASKS")
    print(f"{'=' * 50}")

    patients_labels_dir = os.path.join(dataset_dir, "patients", "labels")
    if os.path.exists(patients_labels_dir):
        patient_masks = sorted(glob(os.path.join(patients_labels_dir, "*.png")))
        if patient_masks:
            print(f"\n--- Patient mask: {os.path.basename(patient_masks[0])} ---")
            patient_mask = cv2.imread(patient_masks[0], cv2.IMREAD_UNCHANGED)

            if patient_mask is not None:
                print(f"Shape: {patient_mask.shape}, dtype: {patient_mask.dtype}")
                print(f"Unique values: {np.unique(patient_mask)}")
                print(f"Min value: {np.min(patient_mask)}")
                print(f"Max value: {np.max(patient_mask)}")

                if len(patient_mask.shape) == 3 and patient_mask.shape[2] == 4:
                    print("Patient mask has alpha channel")
                    print(
                        f"Alpha channel unique values: {np.unique(patient_mask[:, :, 3])}"
                    )


if __name__ == "__main__":
    analyze_mask_files()
