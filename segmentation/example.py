import numpy as np
import base64
from io import BytesIO
from PIL import Image

import wandb
import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import monai


def get_model() -> nn.Module:
    print("Loading model...")
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init()
    artifact = run.use_artifact(
        "nm-i-ki/tumor-segmentation/best_model:v294", type="model"
    )
    artifact_path = artifact.download()
    print(f"Artifact downloaded to: {artifact_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "artifacts/best_model:v294/best_model_0.6785.pth",
            weights_only=True,
        )
    )
    return model


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

    model = get_model()

    # Decode base64 image
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    img_array = np.array(img)

    # Get segmentation
    # segmentation = predict(img_array)
    predicted_segments = model(img_array)

    # Convert back to base64
    seg_img = Image.fromarray(predicted_segments)
    buffered = BytesIO()
    seg_img.save(buffered, format="PNG")
    seg_base64 = base64.b64encode(buffered.getvalue()).decode()

    return seg_base64


### DUMMY MODEL ###
def get_threshold_segmentation(img: np.ndarray, threshold: int) -> np.ndarray:
    return (img < threshold).astype(np.uint8) * 255


if __name__ == "__main__":
    get_model()
