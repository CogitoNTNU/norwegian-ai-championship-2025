"""
tumor-segmentation inference script (ensemble + Albumentations TTA)
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

import cv2  # type: ignore
import numpy as np
import torch
import segmentation_models_pytorch as smp  # type: ignore
from dotenv import load_dotenv
from PIL import Image
import wandb  # type: ignore
import albumentations as A  # type: ignore

# Globals
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Each entry is a single model already moved to _DEVICE and set to eval()
# Lazy-loaded ensemble
_MODELS: List[smp.Unet] | None = None
# Cached paths to downloaded ckpts
_ARTIFACT_DIRS: List[Path] | None = None

_TRAIN_IMG_SIZE: Tuple[int, int] = (1536, 786)  # (H, W)
_DEFAULT_THRESHOLD: float = float(os.getenv("PRED_THRESHOLD", 0.5))


# Test Time Augmentation configuration
def _tta_enabled() -> bool:
    v = (os.getenv("TTA", "1") or "1").strip().lower()
    return v not in {"0", "false", "off", "no"}


def _parse_tta_ops() -> List[str]:
    raw = os.getenv("TTA_OPS", "identity,hflip,vflip,hvflip")
    return [op.strip().lower() for op in raw.split(",") if op.strip()]


class _AlbuTTATransform:
    """Holds an albumentations transform + how to invert it."""

    def __init__(self, name: str, aug: A.Compose):
        self.name = name
        # applied to image at input-time (self-inverse for flips)
        self.aug = aug


def _get_albu_tta_transforms() -> List[_AlbuTTATransform]:
    ops = _parse_tta_ops() if _tta_enabled() else ["identity"]
    ttas: List[_AlbuTTATransform] = []
    for op in ops:
        if op in {"identity", "id"}:
            ttas.append(_AlbuTTATransform("identity", A.Compose([])))
        elif op in {"hflip", "flipx"}:
            ttas.append(
                _AlbuTTATransform("hflip", A.Compose([A.HorizontalFlip(p=1.0)]))
            )
        elif op in {"vflip", "flipy"}:
            ttas.append(_AlbuTTATransform("vflip", A.Compose([A.VerticalFlip(p=1.0)])))
        elif op in {"hvflip", "vhflip", "flipxy"}:
            ttas.append(
                _AlbuTTATransform(
                    "hvflip",
                    A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
                )
            )
        else:
            print(f"[WARN] Unknown TTA op '{op}' – skipping.")
    if not ttas:
        ttas = [_AlbuTTATransform("identity", A.Compose([]))]
    return ttas


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


# Pre/-post-processing
def _resize_and_normalize(img: np.ndarray) -> tuple[np.ndarray, Tuple[int, int]]:
    """
    Returns:
      resized_img: float32 array (H, W, 1) in [0,1]
      original_hw: (H, W) of original image (for final upscaling)
    """
    original_hw = img.shape[:2]

    if img.ndim == 2:
        img = img[..., None]
    img = img.astype("float32")
    # per-image min-max normalization
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn + 1e-8)

    resized = cv2.resize(
        img, (_TRAIN_IMG_SIZE[1], _TRAIN_IMG_SIZE[0]), cv2.INTER_LINEAR
    )
    if resized.ndim == 2:
        resized = resized[..., None]
    return resized.astype("float32"), original_hw


def _postprocess(
    prob_map: np.ndarray, original_hw: Tuple[int, int], thr: float
) -> np.ndarray:
    prob_resized = cv2.resize(
        prob_map, (original_hw[1], original_hw[0]), cv2.INTER_LINEAR
    )
    mask = (prob_resized >= thr).astype(np.uint8) * 255  # 0 or 255
    return np.repeat(mask[:, :, None], 3, axis=2)  # H×W×3 RGB


# Utilities
def _deaugment_np(prob: np.ndarray, op_name: str) -> np.ndarray:
    """Invert the augmentation by flipping back."""
    if op_name in {"identity", "id"}:
        return prob
    if op_name in {"hflip", "flipx"}:
        return np.flip(prob, axis=1)  # flip W
    if op_name in {"vflip", "flipy"}:
        return np.flip(prob, axis=0)  # flip H
    if op_name in {"hvflip", "vhflip", "flipxy"}:
        return np.flip(np.flip(prob, axis=1), axis=0)
    return prob


def _np_to_torch_batch(imgs: List[np.ndarray]) -> torch.Tensor:
    """
    imgs: list of (H, W, 1) float32 in [0,1]
    returns: torch.FloatTensor [N, C=1, H, W] on _DEVICE
    """
    arr = np.stack([im.transpose(2, 0, 1) for im in imgs], axis=0).copy()
    return torch.from_numpy(arr).to(_DEVICE)


def _predict_ensemble_with_albu_tta(resized_img_1c: np.ndarray) -> np.ndarray:
    ttas = _get_albu_tta_transforms()
    aug_imgs = [t.aug(image=resized_img_1c)["image"] for t in ttas]
    batch = _np_to_torch_batch(aug_imgs)
    models = _load_models()
    sum_probs_aug: Optional[np.ndarray] = None  # [N,H,W] across models

    with torch.inference_mode():
        for model in models:
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            sum_probs_aug = probs if sum_probs_aug is None else (sum_probs_aug + probs)

    avg_probs_aug = sum_probs_aug / max(len(models), 1)
    deaug_list = [
        _deaugment_np(avg_probs_aug[i], ttas[i].name) for i in range(len(ttas))
    ]
    return np.mean(np.stack(deaug_list, axis=0), axis=0).astype("float32")


# Public API
def predict_tumor_segmentation(img_data: str) -> str:
    img_bytes = base64.b64decode(img_data)
    np_img = np.array(Image.open(BytesIO(img_bytes)).convert("L"))
    resized_img, orig_hw = _resize_and_normalize(np_img)
    mean_prob_resized = _predict_ensemble_with_albu_tta(resized_img)
    mask_rgb = _postprocess(mean_prob_resized, orig_hw, _DEFAULT_THRESHOLD)
    buf = BytesIO()
    Image.fromarray(mask_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
