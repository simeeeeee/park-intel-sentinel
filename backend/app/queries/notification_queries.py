from typing import List
from app.db.connection import database
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # INFO 이상 출력
# 알림 전체 조회
async def fetch_alert_logs(order: str) -> List[dict]:
    query = f"""
        SELECT 
        max(alert_logs.id) as id,
        max(alert_logs.created_at) as created_at,  
        alert_logs.plate_text, 
        alert_logs.reason,
        max(alert_logs.is_checked) as is_checked,
        parking_zones.name,
        parking_zones.floor
        FROM alert_logs
        LEFT JOIN parking_zones ON alert_logs.zone_id = parking_zones.id
        WHERE alert_logs.deleted_at IS NULL
        AND parking_zones.deleted_at IS NULL
        GROUP BY 
            alert_logs.plate_text, 
            alert_logs.reason,
            parking_zones.name,
            parking_zones.floor
        ORDER BY alert_logs.created_at {order}
    """
    
    logger.info(f"Executing query: {query}")
    return await database.fetch_all(query)


# 알림 삭제
async def delete_alert_log(id: int):
    query = """
        UPDATE alert_logs SET deleted_at = CURRENT_TIMESTAMP WHERE id = :id
    """
    await database.execute(query, {"id": id})


# 알림 조회
# 최근 1분 이내 알림 조회
async def fetch_alert_log_recent(order: str) -> List[dict]:
    query = f"""
        SELECT 
        alert_logs.id,
        alert_logs.created_at,
        alert_logs.plate_text, 
        alert_logs.reason,
        alert_logs.is_checked,
        parking_zones.name,
        parking_zones.floor
        FROM alert_logs
        LEFT JOIN parking_zones ON alert_logs.zone_id = parking_zones.id
        WHERE alert_logs.created_at >= NOW() - INTERVAL 48 HOUR 
        AND alert_logs.deleted_at IS NULL
        AND parking_zones.deleted_at IS NULL
        AND alert_logs.is_checked = FALSE
        ORDER BY alert_logs.created_at {order}
    """
    
    logger.info(f"Executing query: {query}")
    return await database.fetch_all(query)

async def fetch_alert_log(id:int):
    query = """
        SELECT 
        alert_logs.id,
        alert_logs.created_at,
        alert_logs.plate_text, 
        alert_logs.reason,
        alert_logs.is_checked,
        parking_zones.name,
        parking_zones.floor,
        parking_zones.rfid_tag,
        registered_vehicles.entered_at
        FROM alert_logs
        LEFT JOIN parking_zones ON alert_logs.zone_id = parking_zones.id
        LEFT JOIN registered_vehicles ON alert_logs.plate_text = registered_vehicles.plate_text
        LEFT JOIN car_owners ON registered_vehicles.owner_id = car_owners.id
        WHERE alert_logs.id = :id
        AND alert_logs.deleted_at IS NULL
        AND parking_zones.deleted_at IS NULL
    """
    return await database.fetch_one(query, {"id": id})
        
# 알림 is_checked 상태 업데이트
async def update_alert_log_checked(id: int):
    query = """
        UPDATE alert_logs SET is_checked = TRUE WHERE id = :id
    """
    await database.execute(query, {"id": id})
