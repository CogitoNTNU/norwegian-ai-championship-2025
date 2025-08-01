import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from openai import OpenAI

HOST = "0.0.0.0"
PORT = 8000

class MedicalStatementRequestDto(BaseModel):
    statement: str

class MedicalStatementResponseDto(BaseModel):
    statement_is_true: int
    statement_topic: int

app = FastAPI()
start_time = time.time()


# Create OpenAI client pointing to local ollama instance
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="dummy"  # ollama doesn't require real API key
)

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
    statement_is_true = predict_llm(request.statement)
    statement_topic=4


    # Return the prediction
    response = MedicalStatementResponseDto(
        statement_is_true=statement_is_true,
        statement_topic=statement_topic
    )
    logger.info(f'Returning prediction: true={statement_is_true}, topic={statement_topic}')
    return response




    

def predict_llm(statement: str, model_name="llama3.2:3b") -> bool:
    """
    Use local ollama instance with OpenAI-compatible API to determine if a medical statement is true or false.
    
    Args:
        statement (str): The medical statement to evaluate
        model_name (str): The ollama model to use
        
    Returns:
        bool: True if statement is true, False if false
    """
    

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a medical expert. Respond with only 'true' or 'false' to indicate whether the given medical statement is accurate."
                },
                {
                    "role": "user", 
                    "content": f"Is this medical statement true or false?\n\nStatement: {statement}"
                }
            ],
            max_tokens=10,
            temperature=0
        )
        
        output = response.choices[0].message.content.lower().strip()
        
        # Simple parsing to extract true/false
        if "true" in output and "false" not in output:
            return True
        elif "false" in output and "true" not in output:
            return False
        else:
            # Default to false if unclear
            logger.warning(f"Unclear LLM response: {output}")
            return False
            
    except Exception as e:
        logger.error(f"Error calling ollama via OpenAI API: {e}")
        return False


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
