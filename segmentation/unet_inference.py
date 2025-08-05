# tumor‑segmentation inference script
# ==================================
# This file defines the `predict_tumor_segmentation` function that the Docker
# API will call.  It
#   1) downloads the model checkpoint from a Weights & Biases artifact (name is
#      read from the `MODEL_ARTIFACT` environment variable),
#   2) runs a forward‑pass on the input PET MIP, and
#   3) returns a base64‑encoded RGB mask that contains **white** pixels where
#      tumor is present and **black** elsewhere.  The mask has exactly the same
#      height/width as the incoming image.

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple

import cv2  # type: ignore
import numpy as np
import torch
import segmentation_models_pytorch as smp  # type: ignore
from dotenv import load_dotenv
from PIL import Image
import wandb  # type: ignore

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL: smp.Unet | None = None  # lazy‑loaded + cached
_ARTIFACT_DIR: Path | None = None  # cached path to downloaded ckpt
_TRAIN_IMG_SIZE: Tuple[int, int] = (1536, 786)  # (H, W) used during training
_DEFAULT_THRESHOLD: float = float(os.getenv("PRED_THRESHOLD", 0.5))

# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------


def _build_model() -> smp.Unet:
    """Re‑create the exact architecture used during training."""
    model: smp.Unet = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,  # weights come from checkpoint
        in_channels=1,  # PET is single‑channel
        classes=1,  # binary mask
        activation=None,
        decoder_attention_type="scse",
    )
    return model


def _download_artifact() -> Path:
    """Download the checkpoint from W&B the first time we need it."""
    global _ARTIFACT_DIR
    if _ARTIFACT_DIR is not None:
        return _ARTIFACT_DIR

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY must be set in the environment.")

    artifact_name = os.getenv(
        "MODEL_ARTIFACT", "nm-i-ki/tumor-segmentation/best_model:latest"
    )

    # Authenticate once (no‑op if already logged in inside the container)
    wandb.login(key=api_key, relogin=False)

    # Off‑line run just for the artifact download; immediately finished.
    run = wandb.init(project="tumor‑segmentation‑inference", job_type="load-model")
    artifact = run.use_artifact(artifact_name, type="model")
    _ARTIFACT_DIR = Path(artifact.download())
    run.finish()
    return _ARTIFACT_DIR


def _load_model() -> smp.Unet:
    """Return a *cached* instance of the trained model on the correct device."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    ckpt_dir = _download_artifact()

    # Find the first file that looks like a PyTorch checkpoint
    ckpt_path: Path | None = next((p for p in ckpt_dir.rglob("*.pth")), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            "No .pth checkpoint found inside the artifact directory"
        )

    model = _build_model().to(_DEVICE)

    state_dict = torch.load(ckpt_path, map_location=_DEVICE)
    # Handle potential wrappers (Lightning, DDP, etc.)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Strip "module." prefix from DDP checkpoints
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[WARN] load_state_dict issues — missing:",
            missing,
            ", unexpected:",
            unexpected,
        )

    model.eval()
    _MODEL = model
    return _MODEL


# ---------------------------------------------------------------------------
# Pre/‑post‑processing helpers
# ---------------------------------------------------------------------------


def _preprocess(img: np.ndarray) -> tuple[torch.Tensor, Tuple[int, int]]:
    """Convert to float32, min‑max scale, resize to training resolution, ➜ torch.

    The incoming PNG/JPEG can be either H×W (grayscale) **or** H×W×3 (RGB).
    The network was trained on **single‑channel** PET slices, so we always keep
    just one channel.  Crucially, OpenCV drops the channel dimension when
    resizing 1‑channel inputs, so we explicitly add it back if needed.
    """

    # Original spatial size (height, width) so we can later upsample the mask
    original_hw: Tuple[int, int] = img.shape[:2]

    # ------------------------------------------------------------------
    # 1) Ensure a channel dimension exists (H×W×1)
    # ------------------------------------------------------------------
    if img.ndim == 2:  # grayscale H×W
        img = img[..., None]  # → H×W×1

    # ------------------------------------------------------------------
    # 2) Normalise 0‑255 → 0‑1  (same as during training)
    # ------------------------------------------------------------------
    img = img.astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # ------------------------------------------------------------------
    # 3) Resize *while keeping* the channel dim for single‑channel images
    # ------------------------------------------------------------------
    resized = cv2.resize(
        img, (_TRAIN_IMG_SIZE[1], _TRAIN_IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR
    )  # shape may be (H, W) or (H, W, C)

    if resized.ndim == 2:  # channel got stripped → re‑attach
        resized = resized[..., None]

    # ------------------------------------------------------------------
    # 4) HWC → NCHW  (and ensure contiguous memory with .copy())
    # ------------------------------------------------------------------
    tensor = (
        torch.from_numpy(resized.transpose(2, 0, 1).copy())  #  C×H×W, contiguous
        .unsqueeze(0)  # 1×C×H×W
        .to(_DEVICE)
    )

    return tensor, original_hw


def _postprocess(
    prob_map: np.ndarray, original_hw: Tuple[int, int], threshold: float
) -> np.ndarray:
    """Resize back to original size and convert to 0/255 RGB mask."""
    prob_resized = cv2.resize(
        prob_map, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_LINEAR
    )
    mask = (prob_resized > threshold).astype(np.uint8) * 255  # H×W, 0 or 255
    rgb_mask = np.repeat(mask[:, :, None], 3, axis=2)  # H×W×3
    return rgb_mask


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def predict(img: np.ndarray) -> np.ndarray:  # kept for backwards compatibility
    """Dummy threshold baseline ( *NOT* used by the API)."""
    return (img < 50).astype(np.uint8) * 255


def predict_tumor_segmentation(img_data: str) -> str:
    """Entry‑point expected by the scoring server.

    Parameters
    ----------
    img_data : str
        Base64‑encoded PNG/JPEG.  Can be either grayscale or RGB — only the
        first channel is used.

    Returns
    -------
    str
        Base64‑encoded PNG of an RGB mask with shape identical to the input
        image and values either **(255,255,255)** (tumor) or **(0,0,0)**.
    """
    # 1) Decode base64 → numpy
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes)).convert("L")  # force grayscale (1‑chan)
    np_img = np.array(img)

    # 2) Pre‑process & model forward
    tensor, orig_hw = _preprocess(np_img)
    with torch.inference_mode():
        prob = torch.sigmoid(_load_model()(tensor))[0, 0].cpu().numpy()

    # 3) Post‑process → binary RGB mask
    mask_rgb = _postprocess(prob, orig_hw, _DEFAULT_THRESHOLD)

    # 4) Encode back to base64
    buf = BytesIO()
    Image.fromarray(mask_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# CLI helper (debug locally)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python inference.py /path/to/image.png")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    b64_mask = predict_tumor_segmentation(b64)
    print(json.dumps({"mask_base64": b64_mask}))
