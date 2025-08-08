import time
import uvicorn
import datetime
from fastapi import FastAPI
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from unet_tta_inference import predict_tumor_segmentation


HOST = "0.0.0.0"
PORT = 9051

app = FastAPI(
    title="Tumor Segmentation API",
    description="Medical image tumor segmentation",
    version="1.0.0",
)
start_time = time.time()


@app.post("/predict", response_model=TumorPredictResponseDto)
def predict(request: TumorPredictRequestDto):
    segmentation_result = predict_tumor_segmentation(request.img)
    return TumorPredictResponseDto(img=segmentation_result)


@app.get("/api")
def api_info():
    return {
        "service": "tumor-segmentation",
        "version": "1.0.0",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def root():
    return {
        "message": "Tumor Segmentation API",
        "service": "tumor-segmentation",
        "status": "running",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


def start_server():
    """Start the API server with nohup and logging."""
    import subprocess
    import sys
    import os
    import time
    import requests

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Check if server is already running
    try:
        response = requests.get(f"http://localhost:{PORT}/", timeout=2)
        if response.status_code == 200:
            print(f"⚠️  Server already running on http://localhost:{PORT}")
            return None
    except requests.exceptions.RequestException:
        pass  # Server not running, continue

    # Run the API server with nohup and logging
    cmd = [
        "nohup",
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    print("🚀 Starting Tumor Segmentation API...")

    with open("logs/api.log", "w") as log_file:
        process = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=os.getcwd()
        )

    # Wait for server to start and verify it's responding
    print("⏳ Waiting for server to start...")
    for i in range(10):  # Try for 10 seconds
        time.sleep(1)
        try:
            response = requests.get(f"http://localhost:{PORT}/", timeout=2)
            if response.status_code == 200:
                print("✅ Tumor Segmentation API successfully started!")
                print(f"🚀 URL: http://{HOST}:{PORT}")
                print("📝 Logs: logs/api.log")
                print(f"🔍 PID: {process.pid}")
                print(f"⏹️  To stop: kill {process.pid}")
                return process
        except requests.exceptions.RequestException:
            continue

    # If we get here, server didn't start successfully
    print("❌ Failed to start server. Check logs/api.log for details.")
    process.terminate()
    return None


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
