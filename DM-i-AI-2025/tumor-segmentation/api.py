import uvicorn
import time
import datetime
import numpy as np
from fastapi import Body, FastAPI
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from example import predict
from utils import validate_segmentation, encode_request, decode_request


HOST = "0.0.0.0"
PORT = 9051


app = FastAPI()
start_time = time.time()

@app.post('/predict', response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):

    # Decode request str to numpy array
    img: np.ndarray = decode_request(request)

    # Obtain segmentation prediction
    predicted_segmentation = predict(img)

    # Validate segmentation format
    validate_segmentation(img, predicted_segmentation)

    # Encode the segmentation array to a str
    encoded_segmentation = encode_request(predicted_segmentation)

    # Return the encoded segmentation to the validation/evalution service
    response = TumorPredictResponseDto(
        img=encoded_segmentation
    )
    return response


@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"




if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
