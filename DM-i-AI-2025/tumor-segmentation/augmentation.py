import numpy as np
import cv2
from pathlib import Path
import re
import albumentations as A


def create_augmentation_pipeline():
    """
    Create an Albumentations augmentation pipeline for medical images.

    Returns:
        A.Compose: Augmentation pipeline
    """
    return A.Compose(
        [
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            # Intensity transformations (careful with medical images)
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            # Elastic deformation for more realistic variations
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        ]
    )


def create_light_augmentation_pipeline():
    """
    Create a lighter augmentation pipeline for medical images with fewer transformations.

    Returns:
        A.Compose: Light augmentation pipeline
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.05, contrast_limit=0.05, p=0.2
            ),
        ]
    )


def augment_single_pair(
    image: np.ndarray, mask: np.ndarray, transform, num_augmentations: int = 3
):
    """
    Apply augmentations to a single image-mask pair.

    Args:
        image: Input image
        mask: Corresponding segmentation mask
        transform: Albumentations transform pipeline
        num_augmentations: Number of augmented versions to create

    Returns:
        List of tuples (augmented_image, augmented_mask)
    """
    augmented_pairs = []

    for _ in range(num_augmentations):
        augmented = transform(image=image, mask=mask)
        augmented_pairs.append((augmented["image"], augmented["mask"]))

    return augmented_pairs


def extract_number_from_filename(filename: str) -> str:
    """
    Extract the number from a filename.

    Args:
        filename: The filename to extract number from

    Returns:
        The extracted number as string, or None if not found
    """
    # Look for numbers in the filename
    match = re.search(r"(\d+)", filename)
    return match.group(1) if match else None


def find_corresponding_mask(img_filename: str, labels_dir: Path) -> Path:
    """
    Find the corresponding segmentation mask for an image.

    Args:
        img_filename: Name of the image file
        labels_dir: Directory containing label files

    Returns:
        Path to the corresponding mask file, or None if not found
    """
    # Extract number from image filename
    img_number = extract_number_from_filename(img_filename)

    if img_number is None:
        return None

    # Look for segmentation file with the same number
    mask_pattern = f"segmentation_{img_number}.png"
    mask_path = labels_dir / mask_pattern

    if mask_path.exists():
        return mask_path

    # If exact match not found, try to find any segmentation file with the same number
    for mask_file in labels_dir.glob("segmentation_*.png"):
        mask_number = extract_number_from_filename(mask_file.name)
        if mask_number == img_number:
            return mask_file

    return None


def augment_dataset(
    data_dir: Path,
    output_dir: Path,
    num_augmentations_per_image: int = 5,
    pipeline_type: str = "full",
):
    """
    Apply Albumentations augmentation to all images and masks in the dataset.

    Args:
        data_dir: Path to the data directory containing patients folder
        output_dir: Path where augmented data will be saved
        num_augmentations_per_image: Number of augmented versions per original image
        pipeline_type: Type of pipeline to use ("full" or "light")
    """
    patients_dir = data_dir / "patients"
    imgs_dir = patients_dir / "imgs"
    labels_dir = patients_dir / "labels"

    # Validate input directories
    if not imgs_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {imgs_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Create output directories
    output_patients_dir = output_dir / "patients"
    output_imgs_dir = output_patients_dir / "imgs"
    output_labels_dir = output_patients_dir / "labels"

    output_imgs_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Create augmentation pipeline
    if pipeline_type == "light":
        transform = create_light_augmentation_pipeline()
        print("Using light augmentation pipeline")
    else:
        transform = create_augmentation_pipeline()
        print("Using full augmentation pipeline")

    # Get all image files
    image_files = list(imgs_dir.glob("*.png"))

    if not image_files:
        print(f"No PNG files found in {imgs_dir}")
        return

    print(f"Found {len(image_files)} image files")
    print(f"Will create {num_augmentations_per_image} augmentations per image")
    print(
        f"Total files to be created: {len(image_files) * (num_augmentations_per_image + 1)} images and masks"
    )

    successful_pairs = 0
    failed_pairs = 0

    for img_path in image_files:
        try:
            # Load image
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                failed_pairs += 1
                continue

            # Convert BGR to RGB for Albumentations
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Find corresponding mask file
            mask_path = find_corresponding_mask(img_path.name, labels_dir)
            if mask_path is None:
                print(f"Warning: No corresponding mask found for {img_path.name}")
                failed_pairs += 1
                continue

            mask = cv2.imread(
                str(mask_path), cv2.IMREAD_GRAYSCALE
            )  # Load mask as grayscale
            if mask is None:
                print(f"Warning: Could not load mask {mask_path}")
                failed_pairs += 1
                continue

            print(f"Processing: {img_path.name} <-> {mask_path.name}")

            # Copy original files (convert back to BGR for saving)
            original_img_out = output_imgs_dir / img_path.name
            original_mask_out = output_labels_dir / mask_path.name
            cv2.imwrite(str(original_img_out), image)  # Save original in BGR
            cv2.imwrite(str(original_mask_out), mask)

            # Generate augmented versions
            augmented_pairs = augment_single_pair(
                image_rgb, mask, transform, num_augmentations_per_image
            )

            for i, (aug_image, aug_mask) in enumerate(augmented_pairs):
                # Create new filenames
                img_base_name = img_path.stem
                mask_base_name = mask_path.stem
                extension = img_path.suffix

                new_img_name = f"{img_base_name}_aug_{i:02d}{extension}"
                new_mask_name = f"{mask_base_name}_aug_{i:02d}{extension}"

                # Save augmented files (convert back to BGR for image)
                aug_img_path = output_imgs_dir / new_img_name
                aug_mask_path = output_labels_dir / new_mask_name

                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_img_path), aug_image_bgr)
                cv2.imwrite(str(aug_mask_path), aug_mask)

            successful_pairs += 1
            print(
                f"✓ Successfully processed {img_path.name} ({num_augmentations_per_image} augmentations created)"
            )

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            failed_pairs += 1

    # Summary
    print("\n" + "=" * 50)
    print("AUGMENTATION SUMMARY")
    print("=" * 50)
    print(f"Successfully processed: {successful_pairs} image-mask pairs")
    print(f"Failed to process: {failed_pairs} pairs")
    print(
        f"Total augmented images created: {successful_pairs * num_augmentations_per_image}"
    )
    print(
        f"Total files in output directory: {successful_pairs * (num_augmentations_per_image + 1)} each for images and masks"
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Configuration
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "data_augmented"

    if not (data_dir / "patients" / "imgs").exists():
        print(
            "Data directory structure not found. Expected: data/patients/imgs and data/patients/labels"
        )
        exit(1)

    print("Albumentations Data Augmentation Pipeline")
    print("=" * 40)

    # Choose pipeline type
    print("\nChoose augmentation pipeline:")
    print("1. Full pipeline (recommended for training)")
    print("2. Light pipeline (fewer transformations)")

    pipeline_choice = input("Enter choice (1 or 2, default: 1): ").strip()
    pipeline_type = "light" if pipeline_choice == "2" else "full"

    # Choose number of augmentations
    num_augs = input("Number of augmentations per image (default: 5): ").strip()
    num_augs = int(num_augs) if num_augs.isdigit() else 5

    # Confirm before proceeding
    print("\nConfiguration:")
    print(f"- Pipeline type: {pipeline_type}")
    print(f"- Augmentations per image: {num_augs}")
    print(f"- Input directory: {data_dir}")
    print(f"- Output directory: {output_dir}")

    confirm = input("\nProceed with augmentation? (y/n, default: y): ").strip().lower()
    if confirm in ["n", "no"]:
        print("Augmentation cancelled.")
        exit(0)

    # Run augmentation
    try:
        augment_dataset(data_dir, output_dir, num_augs, pipeline_type)
        print(f"\n Augmentation complete! Check {output_dir}")
    except Exception as e:
        print(f"\n Error during augmentation: {str(e)}")
        exit(1)
