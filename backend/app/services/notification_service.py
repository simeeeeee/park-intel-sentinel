from app.queries.notification_queries import *
from app.db.connection import database
from fastapi import Query
from typing import Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # INFO 이상 출력

async def get_notification_list(order: Optional[str] = None):
    try:
        #전체 알림 조회(asc: 오름차순/ desc: 내림차순)
        result = await fetch_alert_logs(order)
        return {"data": result}
    except Exception as e:
            logger.error(f"Error notification_list : {e}")
    
async def get_recent_notification_list(order: Optional[str] = None):
    try:
        #최근 알림 조회(asc: 오름차순/ desc: 내림차순)
        result = await fetch_alert_log_recent(order)
        return {"data": result}
    except Exception as e:
            logger.error(f"Error recent_notification_list : {e}")
            
            
async def delete_notification(id: int):
    try:
        # 알림 삭제
        await delete_alert_log(id)
        return {"message": f"{id} Notification deleted"}
    except Exception as e:
            logger.error(f"Error delete_notification : {e}")