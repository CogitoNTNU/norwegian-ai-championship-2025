from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from task1.main import task1_router
from task2.main import task2_router
from task3.main import task3_router

load_dotenv()


app = FastAPI()
app.include_router(task1_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def main():
    print("")
    return {"Message to you": "Hello from norwegian-ai-championship-2025!"}
