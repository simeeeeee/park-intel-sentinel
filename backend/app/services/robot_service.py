from app.models.robot import RobotStatusRequest
from app.queries.robot_queries import *
from app.db.connection import database
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # INFO 이상 출력

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # 핸들러 레벨도 맞춰야 함

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)



async def process_robot_status(request: RobotStatusRequest):
    robot_id = request.robot_id
    rfid = request.rfid
    vehicles = request.vehicles  # Dict[str, VehicleInfo]

    for key, vehicle in vehicles.items():
        try:
            async with database.transaction():
                # parking_zones 테이블에서 zone_info 조회 (key, rfid 기준)
                zone_id, zone_type = await fetch_parking_zone_info(rfid, key)  # DB 조회 함수
                logger.info(f"Zone for rfid={rfid}, key={key}")
                if zone_id is None:
                    # raise ValueError("존재하지 않는 zone입니다.")
                    logger.info(f"Zone not found for rfid={rfid}, key={key}")
                    continue
                
                # vehicle.text가 비어있으면 해당 zone_id에 대한 alert_logs 테이블에서 삭제
                if not vehicle.text:
                    # alert_logs 테이블에서 zone_id에 대한 모든 데이터 삭제 처리
                    logger.info(f"delete_alert_logs({zone_id})")
                    await delete_alert_logs(zone_id)
                    logger.info(f"Vehicle text is empty for key={key}, skipping")
                    continue
                
                # vehicle.text기반으로 registered_vehicles 테이블에서 vehicle_type 조회
                vehicle_id, vehicle_type = await fetch_vehicle_info(vehicle.text)  # DB 조회 함수
                if vehicle_id is None:
                    # raise ValueError("존재하지 않는 vehicle_type, id입니다.")
                    logger.info(f"vehicle_id not found for {vehicle.text}")
                    continue
                
                if vehicle_type is None:
                    vehicle_type = "unknown"   
                
                if zone_type is None:
                    # raise ValueError("존재하지 않는 zone_type입니다.")
                    logger.info(f"zone_type not found for zone_id {zone_id}")
                    continue
                    
                # zone_id기반으로 alert_logs 테이블에 해당 zone_id에 대한 리스트 조회(단, deleted_at = null 조건)
                logs = await fetch_alert_logs(zone_id)
                for log in logs:
                    # log에서 plate_text가 vehicle.text와 일치하는지 확인
                    if log.plate_text != vehicle.text:
                        # 일치하지 않으면 해당 로그 삭제
                        logger.info(f"delete_alert_log({log.id})")
                        await delete_alert_log(log.id)
                
                # 만약 vehicle_type과 zone_type이 일치하지 않으면, 해당 zone_id에 대한 alert_logs 테이블에 로그 저장
                if zone_type not in ("NORMAL"): 
                    if zone_type != vehicle_type:
                        logger.info(f"vehicle {vehicle_type} / zone {zone_type}")
                        await save_alert_log(zone_id=zone_id, plate_text=vehicle.text, reason=f"vehicle {vehicle_type} / zone {zone_type}")

                
                # robot_logs 테이블에 로그 저장
                # robot_id 없다면 default로 1로 설정
                if robot_id is None:
                    logger.info(f"robot_id is None, setting to default 1")
                    robot_id = 1
                
                
                logger.info(f"save_robot_log({zone_id}, {robot_id}, {rfid}, {vehicle.text})")
                await save_robot_log(zone_id=zone_id, robot_id=robot_id, rfid_tag=rfid, plate_text=vehicle.text)
                logger.info(f"robot_logs 저장 {key}, {vehicle}")
                
        except Exception as e:
            logger.error(f"Error processing vehicle key={key}: {e}")

            
    return {"processed_count": len(vehicles)}



async def get_robot_position(id: int):
    # 로봇 위치 조회 로직
    try:
        async with database.transaction():
            # robot_id기반으로 robot_logs에서 가장 최근의 log를 가져옴
            log = await fetch_robot_log(id)
            
            
            if log is None:
                return {"robot_id": id, "rfid_tag": None, "message": "No log found"}
            
            logger.info(f"robot_logs 조회 {datetime.now()}, 차이 {datetime.now() - timedelta(minutes=1)}")
            if log["created_at"] < (datetime.now() - timedelta(minutes=1)):
                return {"robot_id": id, "rfid_tag": None, "message": "Robot is inactive for more than 1 minute"}

            return {
                "robot_id": id,
                "rfid_tag": log["rfid_tag"],
                "floor": log["floor"],
                "created_at": log["created_at"].isoformat()
            }
            
    except Exception as e:
            logger.error(f"Error robot_position {id}: {e}")

