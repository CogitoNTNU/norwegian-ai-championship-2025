import numpy as np
import base64
from io import BytesIO
from PIL import Image


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(img: np.ndarray) -> np.ndarray:
    threshold = 50
    segmentation = get_threshold_segmentation(img, threshold)
    return segmentation


def predict_tumor_segmentation(img_data: str) -> str:
    """
    Predict tumor segmentation from base64 encoded image data.

    Args:
        img_data: Base64 encoded image data

    Returns:
        Base64 encoded segmentation mask
    """
    # Decode base64 image
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    img_array = np.array(img)

    # Get segmentation
    segmentation = predict(img_array)

    # Convert back to base64
    seg_img = Image.fromarray(segmentation)
    buffered = BytesIO()
    seg_img.save(buffered, format="PNG")
    seg_base64 = base64.b64encode(buffered.getvalue()).decode()

    return seg_base64


### DUMMY MODEL ###
def get_threshold_segmentation(img: np.ndarray, threshold: int) -> np.ndarray:
    return (img < threshold).astype(np.uint8) * 255
