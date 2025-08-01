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

app = FastAPI()
start_time = time.time()

@app.get('/api')
def hello():
    return {
        "service": "emergency-healthcare-rag",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

@app.post('/predict', response_model=MedicalStatementResponseDto)
def predict_endpoint(request: MedicalStatementRequestDto):

    logger.info(f'Received statement: {request.statement[:100]}...')

    # Get prediction from model
    statement_is_true, statement_topic = predict(request.statement)

    # Validate prediction format
    validate_prediction(statement_is_true, statement_topic)

    # Return the prediction
    response = MedicalStatementResponseDto(
        statement_is_true=statement_is_true,
        statement_topic=statement_topic
    )
    logger.info(f'Returning prediction: true={statement_is_true}, topic={statement_topic}')
    return response

if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
