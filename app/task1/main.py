from fastapi import APIRouter

task1_router = APIRouter()


@task1_router.get("/task1")
def task1():
    return {"Task1": "Hello"}
