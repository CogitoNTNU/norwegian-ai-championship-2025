from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.task1.route import task1_router
from src.task2.route import task2_router
from src.task3.route import task3_router

load_dotenv()


app = FastAPI()
app.include_router(task1_router)
app.include_router(task2_router)
app.include_router(task3_router)

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
