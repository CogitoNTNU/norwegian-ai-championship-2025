import numpy as np
import cv2
import base64
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def validate_segmentation(pet_mip, seg_pred):
    assert isinstance(seg_pred, np.ndarray), (
        "Segmentation was not succesfully decoded as a numpy array"
    )
    assert pet_mip.shape == seg_pred.shape, (
        f"Segmentation of shape {seg_pred.shape} is not identical to image shape {pet_mip.shape}"
    )

    unique_vals = list(np.unique(seg_pred))
    allowed_vals = [0, 255]
    unique_vals_str = ", ".join([str(x) for x in (unique_vals)])
    all_values_are_allowed = all(x in allowed_vals for x in unique_vals)
    assert all_values_are_allowed, (
        f"The segmentation contains values {{{unique_vals_str}}} but only values {{0,255}} are allowed"
    )

    assert np.all(seg_pred[:, :, 0] == seg_pred[:, :, 1]) & np.all(
        seg_pred[:, :, 1] == seg_pred[:, :, 2]
    ), "The segmentation values should be identical along the 3 color channels."


def normalize_to_binary(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Normalize pixel values to 0 or 1 based on a threshold.

    Args:
        image: Input image array
        threshold: Threshold value for binarization (default: 0.5)

    Returns:
        Binary image with values 0 or 1
    """
    # Normalize to 0-1 range first if values are outside this range
    if image.max() > 1.0 or image.min() < 0.0:
        normalized = (image - image.min()) / (image.max() - image.min())
    else:
        normalized = image.copy()

    # Apply threshold to get binary values
    binary = (normalized > threshold).astype(np.float32)

    return binary


def dice_score(y_true: np.ndarray, y_pred: np.ndarray):
    y_true_bin = y_true > 0
    y_pred_bin = y_pred > 0
    return 2 * (y_true_bin & y_pred_bin).sum() / (y_true_bin.sum() + y_pred_bin.sum())


def encode_request(np_array: np.ndarray) -> str:
    # Encode the NumPy array as a png image
    success, encoded_img = cv2.imencode(".png", np_array)

    if not success:
        raise ValueError("Failed to encode the image")

    # Convert the encoded image to a base64 string
    base64_encoded_img = base64.b64encode(encoded_img.tobytes()).decode()

    return base64_encoded_img


def decode_request(request) -> np.ndarray:
    encoded_img: str = request.img
    np_img = np.fromstring(base64.b64decode(encoded_img), np.uint8)
    a = cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)
    return a


def plot_prediction(mip, seg, seg_pred):
    score = dice_score(seg, seg_pred)
    print("Dice Score:", dice_score(seg, seg_pred))
    plt.figure(figsize=(9.2, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(mip)
    plt.axis("off")
    plt.title("PET MIP")

    plt.subplot(1, 4, 2)
    plt.imshow(seg)
    plt.axis("off")
    plt.title("True Segmentation")

    plt.subplot(1, 4, 3)
    plt.imshow(seg_pred)
    plt.axis("off")
    plt.title("Predicted Segmentation")

    TP = ((seg_pred > 0) & (seg > 0))[:, :, :1]
    FP = ((seg_pred > 0) & (seg == 0))[:, :, :1]
    FN = ((seg_pred == 0) & (seg > 0))[:, :, :1]
    img = np.concatenate((FP, TP, FN), axis=2).astype(np.uint8) * 255

    plt.subplot(1, 4, 4)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"dice score = {score:.02f}")
    plt.legend(["a", "b"])

    # Create green, red, and blue squares as proxy artists
    green_square = mpatches.Patch(color="green", label="TP")
    red_square = mpatches.Patch(color="red", label="FP")
    blue_square = mpatches.Patch(color="blue", label="FN")

    # Add the proxy artists to the legend
    plt.legend(handles=[green_square, red_square, blue_square], loc="lower right")
    plt.tight_layout(h_pad=2, w_pad=0, pad=1.5)
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    # Try to find an image in the data directory
    data_dir = Path(__file__).parent / "data"

    # Look for image files in patients or controls directories
    image_paths = []
    for subdir in ["patients", "controls"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"]:
                image_paths.extend(list(subdir_path.glob(f"**/{ext}")))

    if not image_paths:
        print("No image files found in data directory.")
        print("Creating a sample image for demonstration...")
        # Create a sample image with various intensity values
        sample_image = np.random.rand(100, 100, 3) * 255  # Random values 0-255

        # Display original and normalized images
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(sample_image.astype(np.uint8))
        plt.title("Original Sample Image\n(Random values 0-255)")
        plt.axis("off")

        # Normalize with default threshold
        normalized_default = normalize_to_binary(sample_image)
        plt.subplot(1, 3, 2)
        plt.imshow(normalized_default, cmap="gray")
        plt.title("Normalized (threshold=0.5)")
        plt.axis("off")

        # Normalize with custom threshold
        normalized_custom = normalize_to_binary(sample_image, threshold=0.3)
        plt.subplot(1, 3, 3)
        plt.imshow(normalized_custom, cmap="gray")
        plt.title("Normalized (threshold=0.3)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        print(
            f"Original image range: {sample_image.min():.2f} - {sample_image.max():.2f}"
        )
        print(
            f"Normalized image range: {normalized_default.min():.2f} - {normalized_default.max():.2f}"
        )

    else:
        # Use the first found image
        image_path = image_paths[0]

        print(f"Loading image: {image_path}")

        # Load the image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {image_path}")
        else:
            # Convert BGR to RGB for proper display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display original and normalized images
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image_rgb)
            plt.title(f"Original Image\n{image_path.name}")
            plt.axis("off")

            # Normalize with default threshold
            normalized_default = normalize_to_binary(image_rgb)
            plt.subplot(1, 3, 2)
            plt.imshow(normalized_default, cmap="gray")
            plt.title("Normalized (threshold=0.5)")
            plt.axis("off")

            # Normalize with custom threshold
            normalized_custom = normalize_to_binary(image_rgb, threshold=0.3)
            plt.subplot(1, 3, 3)
            plt.imshow(normalized_custom, cmap="gray")
            plt.title("Normalized (threshold=0.3)")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            print(f"Original image shape: {image_rgb.shape}")
            print(f"Original image range: {image_rgb.min()} - {image_rgb.max()}")
            print(
                f"Normalized image range: {normalized_default.min():.2f} - {normalized_default.max():.2f}"
            )
            print(f"Unique values in normalized image: {np.unique(normalized_default)}")
