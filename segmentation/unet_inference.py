# tumor-segmentation inference script (ensemble version)
# ======================================================
from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

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

# Each entry is a single model already moved to _DEVICE and set to eval()
_MODELS: List[smp.Unet] | None = None  # lazy-loaded ensemble
_ARTIFACT_DIRS: List[Path] | None = None  # cached paths to downloaded ckpts

_TRAIN_IMG_SIZE: Tuple[int, int] = (1536, 786)  # (H, W)
_DEFAULT_THRESHOLD: float = float(os.getenv("PRED_THRESHOLD", 0.5))


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------
def _build_model() -> smp.Unet:
    """Re-create the exact UNet architecture used during training."""
    return smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,  # weights come from checkpoints
        in_channels=1,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )


def _parse_artifact_names() -> List[str]:
    """Return the list of artifact names to load."""
    names = os.getenv("MODEL_ARTIFACTS") or os.getenv(
        "MODEL_ARTIFACT", "nm-i-ki/tumor-segmentation/best_model:latest"
    )
    return [n.strip() for n in names.split(",") if n.strip()]


def _download_artifacts() -> List[Path]:
    """Download every checkpoint exactly once, return their directories."""
    global _ARTIFACT_DIRS
    if _ARTIFACT_DIRS is not None:
        return _ARTIFACT_DIRS

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY must be set in the environment.")

    wandb.login(key=api_key, relogin=False)

    artifact_dirs: List[Path] = []
    run = wandb.init(project="tumor-segmentation-inference", job_type="load-ensemble")
    for name in _parse_artifact_names():
        artifact = run.use_artifact(name, type="model")
        artifact_dirs.append(Path(artifact.download()))
    run.finish()

    _ARTIFACT_DIRS = artifact_dirs
    return _ARTIFACT_DIRS


def _load_models() -> List[smp.Unet]:
    """Load (and cache) *all* ensemble members on the correct device."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    models: List[smp.Unet] = []
    for ckpt_dir in _download_artifacts():
        ckpt_path = next(ckpt_dir.rglob("*.pth"), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No .pth checkpoint found inside artifact {ckpt_dir}"
            )

        model = _build_model().to(_DEVICE)
        state = torch.load(ckpt_path, map_location=_DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                "[WARN] load_state_dict issues — missing:",
                missing,
                ", unexpected:",
                unexpected,
            )
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("Ensemble list is empty – check MODEL_ARTIFACTS.")
    _MODELS = models
    return _MODELS


# ---------------------------------------------------------------------------
# Pre/-post-processing helpers (unchanged except for typing tweaks)
# ---------------------------------------------------------------------------
def _preprocess(img: np.ndarray) -> tuple[torch.Tensor, Tuple[int, int]]:
    original_hw = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]

    img = img.astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    resized = cv2.resize(
        img, (_TRAIN_IMG_SIZE[1], _TRAIN_IMG_SIZE[0]), cv2.INTER_LINEAR
    )
    if resized.ndim == 2:
        resized = resized[..., None]

    tensor = (
        torch.from_numpy(resized.transpose(2, 0, 1).copy())
        .unsqueeze(0)  # 1×C×H×W
        .to(_DEVICE)
    )
    return tensor, original_hw


def _postprocess(
    prob_map: np.ndarray, original_hw: Tuple[int, int], thr: float
) -> np.ndarray:
    prob_resized = cv2.resize(
        prob_map, (original_hw[1], original_hw[0]), cv2.INTER_LINEAR
    )
    mask = (prob_resized >= thr).astype(np.uint8) * 255  # 0 or 255
    return np.repeat(mask[:, :, None], 3, axis=2)  # H×W×3 RGB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_tumor_segmentation(img_data: str) -> str:
    """Main entry-point for the scoring server (ensemble version)."""
    img_bytes = base64.b64decode(img_data)
    np_img = np.array(Image.open(BytesIO(img_bytes)).convert("L"))  # grayscale

    tensor, orig_hw = _preprocess(np_img)
    with torch.inference_mode():
        probs = [
            torch.sigmoid(model(tensor))[0, 0]  # C=1 ⇒ pick [0,0]
            for model in _load_models()
        ]
        mean_prob = torch.stack(probs).mean(dim=0).cpu().numpy()

    mask_rgb = _postprocess(mean_prob, orig_hw, _DEFAULT_THRESHOLD)

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
