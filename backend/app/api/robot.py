from fastapi import APIRouter, Depends, HTTPException
from app.models.robot import RobotStatusRequest
from app.services.robot_service import *

router = APIRouter()

@router.post("/status")
async def robot_status(request: RobotStatusRequest):
    try:
        result = await process_robot_status(request)
        return {"result": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def robot_position(id: int):
    try:
        result = await get_robot_position(id)
        return {"result": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
