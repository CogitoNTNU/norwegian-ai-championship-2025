import time
import uvicorn
import datetime
from fastapi import Body, FastAPI, Request
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from example import predict_race_car_action
from src.game.core import determine_lane_change_direction

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(
    title="AI Race Car Challenge",
    description="AI race car control system",
    version="1.0.0",
    debug=True,
)
start_time = time.time()


def predict_actions_from_game_bot(request_data: dict) -> list:
    """
    Use the game bot logic from core.py to predict actions based on sensor data.
    Returns a list of actions based on the current game state.
    """
    # Extract sensor data
    sensors = request_data.get("sensors", {})
    velocity = request_data.get("velocity", {"x": 10, "y": 0})
    did_crash = request_data.get("did_crash", False)

    # If crashed, return nothing
    if did_crash:
        return ["NOTHING"]

    # Get front sensor reading (assuming sensor_0 or 'front' is the front sensor)
    front_sensor = sensors.get("sensor_0") or sensors.get("front") or sensors.get("0")
    back_sensor = sensors.get("sensor_4") or sensors.get("back") or sensors.get("4")

    # Simulate the core.py bot logic
    actions = []

    # Check if there's an obstacle in front
    if front_sensor is not None and front_sensor < 999:  # Obstacle detected in front
        # Determine if we should change lanes based on left/right sensors
        left_sensors = []
        right_sensors = []

        # Collect left and right sensor readings
        for sensor_key, reading in sensors.items():
            if reading is not None:
                if "left" in str(sensor_key).lower():
                    left_sensors.append(reading)
                elif "right" in str(sensor_key).lower():
                    right_sensors.append(reading)

        # Calculate sums for lane change decision
        left_sum = sum(left_sensors) if left_sensors else 0
        right_sum = sum(right_sensors) if right_sensors else 0
        min_safe_distance = 340
        higher_threshold_sensors = [
            "front_right_front",
            "back_right_back",
            "back_left_back",
            "front_left_front",
        ]

        # Check if lanes are blocked with different thresholds
        left_blocked = False
        right_blocked = False

        for sensor_key, reading in sensors.items():
            if reading is not None:
                # Use higher threshold for specific sensors
                threshold = (
                    500
                    if str(sensor_key) in higher_threshold_sensors
                    else min_safe_distance
                )

                if "left" in str(sensor_key).lower() and reading < threshold:
                    left_blocked = True
                elif "right" in str(sensor_key).lower() and reading < threshold:
                    right_blocked = True

        if left_blocked and right_blocked:
            # Both directions blocked - slow down
            actions.extend(["DECELERATE"])
        elif left_blocked:
            # Left blocked, go right if possible
            actions.extend(["STEER_RIGHT"] * 48)  # First phase
            actions.extend(["STEER_LEFT"] * 47)  # Second phase to straighten
        elif right_blocked:
            # Right blocked, go left if possible
            actions.extend(["STEER_LEFT"] * 48)  # First phase
            actions.extend(["STEER_RIGHT"] * 47)  # Second phase to straighten
        elif left_sum > right_sum:
            # Left is clearer, go left
            actions.extend(["STEER_LEFT"] * 48)
            actions.extend(["STEER_RIGHT"] * 47)
        else:
            # Right is clearer or equal, go right
            actions.extend(["STEER_RIGHT"] * 48)
            actions.extend(["STEER_LEFT"] * 47)

    # Check back sensor for approaching cars
    elif back_sensor is not None and back_sensor < 800:  # Car approaching from behind
        # Similar lane change logic but more urgent
        left_sensors = []
        right_sensors = []

        for sensor_key, reading in sensors.items():
            if reading is not None:
                if "left" in str(sensor_key).lower():
                    left_sensors.append(reading)
                elif "right" in str(sensor_key).lower():
                    right_sensors.append(reading)

        left_sum = sum(left_sensors) if left_sensors else 0
        right_sum = sum(right_sensors) if right_sensors else 0

        if left_sum > right_sum:
            actions.extend(["STEER_LEFT"] * 48)
            actions.extend(["STEER_RIGHT"] * 47)
        else:
            actions.extend(["STEER_RIGHT"] * 48)
            actions.extend(["STEER_LEFT"] * 47)

    # Default behavior - accelerate if no immediate threats
    if not actions:
        current_speed = abs(velocity.get("x", 10))
        if current_speed < 23:  # Target speed
            actions.append("ACCELERATE")
        else:
            actions.append("NOTHING")

    # Ensure we return at least one action and it's always a list
    if not actions:
        actions = ["ACCELERATE"]

    return actions


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    print(f"Request body: {body}")
    # Reset the request body for the endpoint to read
    request._body = body
    response = await call_next(request)
    return response


@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    # Use the game bot logic from core.py
    actions = predict_actions_from_game_bot(request.dict())

    # Ensure actions is always a list
    if not isinstance(actions, list):
        actions = [actions]

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
