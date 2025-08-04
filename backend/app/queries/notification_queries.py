from typing import List
from app.db.connection import database

# 알림 전체 조회
async def fetch_alert_logs(order: str) -> List[dict]:
    order = order.lower()
    if order not in ('asc', 'desc'):
        order = 'desc'  # 기본값

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
        WHERE alert_logs.deleted_at IS NULL
        AND parking_zones.deleted_at IS NULL
        ORDER BY created_at {order}
    """
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
    order = order.lower()
    if order not in ('asc', 'desc'):
        order = 'desc'  # 기본값 설정
        
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
        WHERE created_at >= NOW() - INTERVAL 1 MINUTE
        AND alert_logs.deleted_at IS NULL
        AND parking_zones.deleted_at IS NULL
        AND is_checked = FALSE
        ORDER BY created_at {order}
    """
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
        parking_zones.floor
        FROM alert_logs
        LEFT JOIN parking_zones ON alert_logs.zone_id = parking_zones.id
        WHERE created_at >= NOW() - INTERVAL 1 MINUTE
        AND alert_logs.deleted_at IS NULL
        AND parking_zones.deleted_at IS NULL
        AND is_checked = FALSE
    """
    return await database.fetch_one(query, {"id": id})\
        
# 알림 is_checked 상태 업데이트
async def update_alert_log_checked(id: int):
    query = """
        UPDATE alert_logs SET is_checked = TRUE WHERE id = :id
    """
    await database.execute(query, {"id": id})
