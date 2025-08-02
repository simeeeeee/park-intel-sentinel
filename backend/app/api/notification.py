from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from app.services.notification_service import *


router = APIRouter()

@router.get("/")
async def notification_list(order: Optional[str] = Query(default=None)):
    try:
        result = await get_notification_list(order)
        return {"result": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def recent_notification_list(order: Optional[str] = Query(default=None)):
    try:
        result = await get_recent_notification_list(order)
        return {"result": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/")
async def delete_alert(id: int):
    try:
        result = await delete_notification(id)
        return {"result": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
