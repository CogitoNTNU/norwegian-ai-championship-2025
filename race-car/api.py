import time
import uvicorn
import datetime
from fastapi import Body, FastAPI
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from example import predict_race_car_action

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(
    title="AI Race Car Challenge",
    description="AI race car control system",
    version="1.0.0",
)
start_time = time.time()


@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    actions = predict_race_car_action(request.dict())
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


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
