import uvicorn
from fastapi import FastAPI
import datetime
import time
from utils import validate_prediction
from model import predict
from loguru import logger
from pydantic import BaseModel

HOST = "0.0.0.0"
PORT = 8000


class MedicalStatementRequestDto(BaseModel):
    statement: str


class MedicalStatementResponseDto(BaseModel):
    statement_is_true: int
    statement_topic: int


app = FastAPI(
    title="Emergency Healthcare RAG API",
    description="Medical statement classification and topic identification",
    version="1.0.0",
)
start_time = time.time()


@app.get("/api")
def api_info():
    return {
        "service": "emergency-healthcare-rag",
        "version": "1.0.0",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def root():
    return {
        "message": "Emergency Healthcare RAG API",
        "service": "emergency-healthcare-rag",
        "status": "running",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.post("/predict", response_model=MedicalStatementResponseDto)
def predict_endpoint(request: MedicalStatementRequestDto):
    logger.info(f"Received statement: {request.statement[:100]}...")

    # Get prediction from model
    statement_is_true, statement_topic = predict(request.statement)

    # Validate prediction format
    validate_prediction(statement_is_true, statement_topic)

    # Return the prediction
    response = MedicalStatementResponseDto(
        statement_is_true=statement_is_true, statement_topic=statement_topic
    )
    logger.info(
        f"Returning prediction: true={statement_is_true}, topic={statement_topic}"
    )
    return response


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

    print("🚀 Starting Emergency Healthcare RAG API...")

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
                print("✅ Emergency Healthcare RAG API successfully started!")
                print(f"🏥 URL: http://{HOST}:{PORT}")
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
