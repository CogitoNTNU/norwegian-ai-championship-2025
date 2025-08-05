import time
import uvicorn
import datetime
from fastapi import Body, FastAPI, Request
from rule_based_AI.bot_with_memory import LaneChangeController
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(
    title="AI Race Car Challenge",
    description="AI race car control system",
    version="1.0.0",
    debug=True,
)
start_time = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    print(f"Request body: {body}")
    # Reset the request body for the endpoint to read
    request._body = body
    response = await call_next(request)
    return response

controlelr = LaneChangeController()

@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    # Use the game bot logic from core.py
    #actions = predict_actions_from_game_bot(request.model_dump())
    actions = controlelr.predict_actions(request.model_dump())

    return RaceCarPredictResponseDto(actions=actions)


@app.get("/api")
def api_info():
    return {
        "service": "race-car",
        "version": "1.0.0",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def root():
    return {
        "message": "AI Race Car Challenge API",
        "service": "race-car",
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
        "--reload",
    ]

    print("🚀 Starting Race Car API...")

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
                print("✅ Race Car API successfully started!")
                print(f"🏁 URL: http://{HOST}:{PORT}")
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
