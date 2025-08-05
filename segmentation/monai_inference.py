"""
Tumor-segmentation inference (pads to /16, binary RGB mask 0/255).
"""

import os
import base64
from io import BytesIO
from dotenv import load_dotenv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import wandb


_MODEL: nn.Module | None = None


def get_model() -> nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    print("Loading model …")
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init()
    art_dir = run.use_artifact(
        "nm-i-ki/tumor-segmentation/best_model:v294", type="model"
    ).download()

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
            os.path.join(art_dir, "best_model_0.6785.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    _MODEL = model
    return model


def _preprocess(pil: Image.Image, device) -> torch.Tensor:
    arr = np.transpose(np.array(pil.convert("RGBA"), np.float32) / 255.0, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device)  # 1,4,H,W


def _pad16(t: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = t.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    return F.pad(t, (0, pw, 0, ph), mode="reflect"), ph, pw


def _uncrop(y: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    if ph:
        y = y[..., :-ph, :]
    if pw:
        y = y[..., :, :-pw]
    return y


def _postprocess(logits: torch.Tensor) -> Image.Image:
    mask = (torch.argmax(logits, 1).squeeze(0) != 0).cpu().numpy().astype(
        np.uint8
    ) * 255
    mask_rgb = np.repeat(mask[..., None], 3, axis=-1).astype(
        np.uint8
    )  # ensure exact 0/255
    return Image.fromarray(mask_rgb, mode="RGB")


def predict_tumor_segmentation(img_b64: str) -> str:
    model = get_model()
    device = next(model.parameters()).device

    pil = Image.open(BytesIO(base64.b64decode(img_b64)))

    x = _preprocess(pil, device)  # CHANGED (split for clarity)
    x_pad, ph, pw = _pad16(x)  # CHANGED (correct unpacking)

    with torch.no_grad():
        y = model(x_pad)

    y = _uncrop(y, ph, pw)
    mask_img = _postprocess(y)

    buf = BytesIO()
    mask_img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()
