from fastapi import APIRouter

task1_router = APIRouter()


@task1_router.post("/task1")
def task1():
    return {"Task1": "Hello"}
