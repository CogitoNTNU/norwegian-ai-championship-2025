import time
import uvicorn
import datetime
from fastapi import FastAPI
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from example import predict_tumor_segmentation

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


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
