from typing import Optional, Tuple, List
from app.db.connection import database  # DB 연결 인스턴스

# zone_id 조회
async def fetch_zone_id(rfid: str, zone_name: str) -> Optional[int]:
    query = """
        SELECT id FROM parking_zones
        WHERE rfid_tag = :rfid AND name = :zone_name
    """
    result = await database.fetch_one(query, {"rfid": rfid, "zone_name": zone_name})
    return result["id"] if result else None

# zone_type 조회
async def fetch_zone_type(zone_id: int) -> Optional[str]:
    query = "SELECT zone_type FROM parking_zones WHERE id = :zone_id"
    result = await database.fetch_one(query, {"zone_id": zone_id})
    return result["zone_type"] if result else None

# 차량 번호 기반 vehicle_type, vehicle_id 조회
async def fetch_vehicle_type(plate_text: str) -> Tuple[Optional[str], Optional[int]]:
    query = """
        SELECT id, vehicle_type FROM registered_vehicles
        WHERE plate_text = :plate_text
    """
    result = await database.fetch_one(query, {"plate_text": plate_text})
    if result:
        return result["vehicle_type"], result["id"]
    return None, None

# alert_logs 조회
async def fetch_alert_logs(zone_id: int) -> List[dict]:
    query = """
        SELECT id, plate_text FROM alert_logs
        WHERE zone_id = :zone_id AND deleted_at IS NULL
    """
    return await database.fetch_all(query, {"zone_id": zone_id})

# alert_log 삭제
async def delete_alert_log(log_id: int):
    query = """
        UPDATE alert_logs SET deleted_at = CURRENT_TIMESTAMP WHERE id = :log_id
    """
    await database.execute(query, {"log_id": log_id})

# alert_log 전체 삭제 (해당 zone_id 기준)
async def delete_alert_logs(zone_id: int):
    query = """
        UPDATE alert_logs SET deleted_at = CURRENT_TIMESTAMP WHERE zone_id = :zone_id AND deleted_at IS NULL
    """
    await database.execute(query, {"zone_id": zone_id})

# alert_log 저장
async def save_alert_log(zone_id: int, plate_text: str, reason: str):
    query = """
        INSERT INTO alert_logs (zone_id, plate_text, reason, created_at)
        VALUES (:zone_id, :plate_text, :reason, CURRENT_TIMESTAMP)
    """
    await database.execute(query, {
        "zone_id": zone_id,
        "plate_text": plate_text,
        "reason": reason
    })

# robot_log 저장
async def save_robot_log(zone_id: int, robot_id: int, rfid_tag: str, plate_text: str):
    query = """
        INSERT INTO robot_logs (zone_id, robot_id, rfid_tag, plate_text, created_at)
        VALUES (:zone_id, :robot_id, :rfid_tag, :plate_text, CURRENT_TIMESTAMP)
    """
    await database.execute(query, {
        "zone_id": zone_id,
        "rfid_tag": rfid_tag,
        "plate_text": plate_text
    })


async def fetch_robot_log(robot_id: int):
    query = """
        SELECT *
        FROM robot_logs
        WHERE robot_id = :robot_id
        ORDER BY created_at DESC
        LIMIT 1
    """
    await database.fetch_one(query, {"robot_id": robot_id})