import base64
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image

API_URL = "http://localhost:9051/predict"
IMG_PATH = Path(
    "/home/svernys/Documents/cogito/norwegian-ai-championship-2025/data/raw/tumor-segmentation/patients/imgs/patient_000.png"
)
OUT_PATH = Path("patient_000_mask.png")


def encode_png_to_b64(p: Path) -> str:
    # read as-is; your API converts to grayscale internally
    with p.open("rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("utf-8")


def decode_b64_png_to_file(b64_str: str, out_path: Path) -> None:
    data = base64.b64decode(b64_str)
    img = Image.open(BytesIO(data))
    img.save(out_path)


def main():
    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    b64_img = encode_png_to_b64(IMG_PATH)

    payload = {"img": b64_img}  # TumorPredictRequestDto expects {img: str}
    resp = requests.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()

    # body should match TumorPredictResponseDto: {img: <base64 png>}
    mask_b64 = body["img"]
    decode_b64_png_to_file(mask_b64, OUT_PATH)
    print(f"✅ Wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
