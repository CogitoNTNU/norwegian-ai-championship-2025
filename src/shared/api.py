"""
Unified API for Norwegian AI Championship 2025
Handles Emergency Healthcare RAG, Tumor Segmentation, and Race Car Control tasks.
"""

import uvicorn
import time
import datetime
import os
import sys
import signal
import subprocess
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Dict, List, Optional
from loguru import logger

# tomllib is available in Python 3.11+ for TOML parsing if needed

# Host and port settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# FastAPI app
app = FastAPI(
    title="Norwegian AI Championship 2025 - Multi-Task API",
    description="Unified API for Emergency Healthcare RAG, Tumor Segmentation, and Race Car Control",
    version="1.0.0",
)
start_time = time.time()

# =============================================================================
# Data Transfer Objects (DTOs)
# =============================================================================


# Emergency Healthcare RAG DTOs
class MedicalStatementRequestDto(BaseModel):
    statement: str


class MedicalStatementResponseDto(BaseModel):
    statement_is_true: int
    statement_topic: int


# Tumor Segmentation DTOs
class TumorPredictRequestDto(BaseModel):
    img: str


class TumorPredictResponseDto(BaseModel):
    img: str


# Race Car Control DTOs
class RaceCarPredictRequestDto(BaseModel):
    did_crash: bool
    elapsed_time_ms: int
    distance: int
    velocity: Dict[str, int]
    coordinates: Dict[str, int]
    sensors: Dict[str, Optional[int]]


class RaceCarPredictResponseDto(BaseModel):
    actions: List[str]


# =============================================================================
# Prediction Functions (Placeholder implementations)
# =============================================================================


def predict_medical_statement(statement: str) -> tuple[int, int]:
    """
    Predict if a medical statement is true and its topic.

    Args:
        statement: Medical statement to evaluate

    Returns:
        Tuple of (statement_is_true, statement_topic)
    """
    # TODO: Implement your RAG prediction logic here
    # This is a placeholder implementation
    logger.info(f"Processing medical statement: {statement[:50]}...")

    # Example logic - replace with your actual implementation
    statement_is_true = 1 if "true" in statement.lower() else 0
    statement_topic = 0  # Default topic

    return statement_is_true, statement_topic


def predict_tumor_segmentation(img_data: str) -> str:
    """
    Predict tumor segmentation from image data.

    Args:
        img_data: Base64 encoded image data

    Returns:
        Base64 encoded segmentation mask
    """
    # TODO: Implement your segmentation logic here
    # This is a placeholder implementation
    logger.info("Processing tumor segmentation request...")

    # Placeholder - return the input image as segmentation
    # Replace with your actual segmentation model
    return img_data


def predict_race_car_action(game_state: Dict) -> List[str]:
    """
    Predict race car actions based on game state.

    Args:
        game_state: Current game state data

    Returns:
        List of actions to take
    """
    # TODO: Implement your race car control logic here
    # This is a placeholder implementation
    logger.info("Processing race car control request...")

    # Example logic - replace with your actual implementation
    if game_state.get("did_crash", False):
        return ["NOTHING"]

    # Simple logic: accelerate if not crashed
    return ["ACCELERATE"]


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Norwegian AI Championship 2025 - Multi-Task API",
        "tasks": ["emergency-healthcare-rag", "tumor-segmentation", "race-car"],
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/api")
def api_info():
    """API information endpoint."""
    return {
        "service": "multi-task-api",
        "version": "1.0.0",
        "tasks": {
            "emergency-healthcare-rag": "/healthcare/predict",
            "tumor-segmentation": "/tumor/predict",
            "race-car": "/racecar/predict",
        },
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


# Emergency Healthcare RAG endpoint
@app.post("/healthcare/predict", response_model=MedicalStatementResponseDto)
def predict_healthcare(request: MedicalStatementRequestDto):
    """
    Predict if a medical statement is true and classify its topic.
    """
    print(f"\n🩺 [RAG] Received request: {request.statement[:100]}...")
    logger.info(f"Healthcare RAG request: {request.statement[:100]}...")

    try:
        statement_is_true, statement_topic = predict_medical_statement(
            request.statement
        )

        response = MedicalStatementResponseDto(
            statement_is_true=statement_is_true, statement_topic=statement_topic
        )

        print(
            f"✅ [RAG] Response: statement_is_true={statement_is_true}, statement_topic={statement_topic}"
        )
        logger.info(
            f"Healthcare response: true={statement_is_true}, topic={statement_topic}"
        )
        return response

    except Exception as e:
        print(f"❌ [RAG] Error: {e}")
        logger.error(f"Healthcare prediction error: {e}")
        raise


# Tumor Segmentation endpoint
@app.post("/tumor/predict", response_model=TumorPredictResponseDto)
def predict_tumor(request: TumorPredictRequestDto):
    """
    Predict tumor segmentation from medical image.
    """
    logger.info("Tumor segmentation request received")

    try:
        segmentation_result = predict_tumor_segmentation(request.img)

        response = TumorPredictResponseDto(img=segmentation_result)

        logger.info("Tumor segmentation completed")
        return response

    except Exception as e:
        logger.error(f"Tumor segmentation error: {e}")
        raise


# Race Car Control endpoint
@app.post("/racecar/predict", response_model=RaceCarPredictResponseDto)
def predict_racecar(request: RaceCarPredictRequestDto = Body(...)):
    """
    Predict race car actions based on current game state.
    """
    logger.info("Race car control request received")

    try:
        actions = predict_race_car_action(request.dict())

        response = RaceCarPredictResponseDto(actions=actions)

        logger.info(f"Race car actions: {actions}")
        return response

    except Exception as e:
        logger.error(f"Race car prediction error: {e}")
        raise


# Legacy compatibility endpoints (maintain original URLs)
@app.post("/predict", response_model=MedicalStatementResponseDto)
def predict_legacy_healthcare(request: MedicalStatementRequestDto):
    """Legacy endpoint for healthcare predictions."""
    return predict_healthcare(request)


# =============================================================================
# Application Entry Point
# =============================================================================


def kill_process_on_port(port: int):
    """Kill any process running on the specified port."""
    import subprocess

    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    try:
                        logger.info(f"Killing process {pid} on port {port}")
                        subprocess.run(["kill", "-9", pid], timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Timeout killing process {pid}")
                    except Exception as e:
                        logger.warning(f"Could not kill process {pid}: {e}")

            # Give it a moment to clean up
            import time

            time.sleep(1)

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout checking for processes on port {port}")
    except FileNotFoundError:
        # lsof not available, skip cleanup
        pass
    except Exception as e:
        logger.warning(f"Could not check/kill processes on port {port}: {e}")


def create_app_for_task(task: str):
    """Create a FastAPI app for a specific task."""
    if task == "rag":
        task_app = FastAPI(
            title="Emergency Healthcare RAG API",
            description="Medical statement classification and topic identification",
            version="1.0.0",
        )

        @task_app.get("/")
        def rag_root():
            return {"service": "emergency-healthcare-rag", "status": "running"}

        @task_app.get("/api")
        def rag_api():
            return {"service": "emergency-healthcare-rag", "version": "1.0.0"}

        @task_app.post("/predict", response_model=MedicalStatementResponseDto)
        def rag_predict(request: MedicalStatementRequestDto):
            return predict_healthcare(request)

    elif task == "segmentation":
        task_app = FastAPI(
            title="Tumor Segmentation API",
            description="Medical image tumor segmentation",
            version="1.0.0",
        )

        @task_app.get("/")
        def seg_root():
            return {"service": "tumor-segmentation", "status": "running"}

        @task_app.get("/api")
        def seg_api():
            return {"service": "tumor-segmentation", "version": "1.0.0"}

        @task_app.post("/predict", response_model=TumorPredictResponseDto)
        def seg_predict(request: TumorPredictRequestDto):
            return predict_tumor(request)

    elif task == "racecar":
        task_app = FastAPI(
            title="Race Car Control API",
            description="AI race car control system",
            version="1.0.0",
        )

        @task_app.get("/")
        def car_root():
            return {"service": "race-car", "status": "running"}

        @task_app.get("/api")
        def car_api():
            return {"service": "race-car", "version": "1.0.0"}

        @task_app.post("/predict", response_model=RaceCarPredictResponseDto)
        def car_predict(request: RaceCarPredictRequestDto):
            return predict_racecar(request)

    return task_app


def start_multi_port_server():
    """Start the API on three different ports simultaneously."""
    import threading

    # Configuration for each service
    services = [
        {"task": "rag", "port": 8000, "name": "Emergency Healthcare RAG"},
        {"task": "segmentation", "port": 9051, "name": "Tumor Segmentation"},
        {"task": "racecar", "port": 9052, "name": "Race Car Control"},
    ]

    threads = []

    for service in services:
        # Kill any existing process on this port
        logger.info(f"Checking for existing processes on port {service['port']}...")
        kill_process_on_port(service["port"])

        # Create app for this task
        task_app = create_app_for_task(service["task"])

        # Start server in a thread
        def run_server(app, port, name):
            logger.info(f"Starting {name} on {HOST}:{port}")
            uvicorn.run(app, host=HOST, port=port, log_level="info")

        thread = threading.Thread(
            target=run_server, args=(task_app, service["port"], service["name"])
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

        # Small delay between starts
        time.sleep(0.5)

    logger.info("🚀 All services started!")
    logger.info("   - Emergency Healthcare RAG: http://localhost:8000")
    logger.info("   - Tumor Segmentation: http://localhost:9051")
    logger.info("   - Race Car Control: http://localhost:9052")

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down all services...")


def main():
    """Main entry point for the API."""
    start_multi_port_server()


# =============================================================================
# Background Management Functions
# =============================================================================


def start_background():
    """Start API in the background."""
    import subprocess
    import os

    # Kill any existing process on port 8000
    kill_process_on_port(8000)

    # Start the API in background using nohup
    cmd = f"nohup {sys.executable} -c 'from api import main; main()' > api.log 2>&1 &"
    subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

    # Give it a moment to start
    time.sleep(2)

    # Check if it's actually running by testing the port
    import requests

    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ API started successfully and running on http://localhost:8000")
            print("🔄 Check status with: uv run api-status")
        else:
            print("⚠️ API started but may not be responding correctly")
    except requests.exceptions.RequestException:
        print("❌ API failed to start or is not responding")
        print("📝 Check api.log for details")


def stop_background():
    """Stop the background API process."""
    try:
        with open(".api_pid", "r") as f:
            pid = int(f.read())
        os.kill(pid, signal.SIGTERM)
        print("🛑 API stopped")
    except Exception as e:
        print(f"Error stopping API: {e}")


def api_status():
    """Check status of the background API process."""
    try:
        with open(".api_pid", "r") as f:
            pid = int(f.read())
        process = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True)
        if str(pid) in process.stdout:
            print(f"✅ API is running (PID: {pid})")
        else:
            print("⚠️ API is not running")
    except Exception:
        print("⚠️ API is not running")


if __name__ == "__main__":
    main()
