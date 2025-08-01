from fastapi import FastAPI
from app.db.connection import database
from app.api import robot
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


app = FastAPI()

@app.get("/api/ping")
def ping():
    return {"message": "pong"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# 라우터 등록
app.include_router(robot.router, prefix="/api/robot", tags=["Robot"])
