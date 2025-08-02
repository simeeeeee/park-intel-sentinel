from typing import List
from app.db.connection import database

# 알림 전체 조회
async def fetch_alert_logs(order: str) -> List[dict]:
    order = order.lower()
    if order not in ('asc', 'desc'):
        order = 'asc'  # 기본값

    query = f"""
        SELECT * 
        FROM alert_logs
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
        SELECT * 
        FROM alert_logs
        WHERE created_at >= NOW() - INTERVAL 1 MINUTE
        ORDER BY created_at {order}
    """
    return await database.fetch_all(query)