import numpy as np


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(img: np.ndarray) -> np.ndarray:
    threshold = 50
    segmentation = get_threshold_segmentation(img, threshold)
    return segmentation


### DUMMY MODEL ###
def get_threshold_segmentation(img: np.ndarray, threshold: int) -> np.ndarray:
    return (img < threshold).astype(np.uint8) * 255
