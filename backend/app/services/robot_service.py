from app.models.robot import RobotStatusRequest
# from app.queries import robot_queries
from app.queries.robot_queries import *
from app.db.connection import database
import logging

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
                # parking_zones 테이블에서 zone_id 조회 (key, rfid 기준)
                zone_id = await fetch_zone_id(rfid, key)  # DB 조회 함수
                if zone_id is None:
                    # raise ValueError("존재하지 않는 zone입니다.")
                    logger.info(f"Zone not found for rfid={rfid}, key={key}")
                    continue
                
                # 만약 zone_type이 normal인 경우, vehicle.text가 비어있으면 해당 zone_id에 대한 alert_logs 테이블에서 삭제
                if not vehicle.text:
                    # alert_logs 테이블에서 zone_id에 대한 모든 데이터 삭제 처리
                    logger.info(f"delete_alert_logs({zone_id})")
                    await delete_alert_logs(zone_id)
                    continue
                
                # vehicle.text기반으로 registered_vehicles 테이블에서 vehicle_type 조회
                vehicle_type, vehicle_id = await fetch_vehicle_type(vehicle.text)  # DB 조회 함수
                if vehicle_id is None:
                    # raise ValueError("존재하지 않는 vehicle_type, id입니다.")
                    logger.info(f"vehicle_id not found for {vehicle.text}")
                    continue
                
                if vehicle_type is None:
                    vehicle_type = "unknown"   
                
                # zone_id기반으로 해당 zone의 zone_type 조회
                zone_type = await fetch_zone_type(zone_id)  # DB 조회 함수
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
                if vehicle_type != zone_type: 
                    #장애인등록 - 전기차구역은 가능
                    if vehicle_type == "DISABLED" and zone_type == "EV":
                        logger.info(f"장애인 등록- 전기차구역 vehicle {vehicle_type} / zone {zone_type}")
                        continue
 
                    await save_alert_log(zone_id=zone_id, plate_text=vehicle.text, reason=f"vehicle {vehicle_type} / zone {zone_type}")
                
                # robot_logs 테이블에 로그 저장
                # robot_id 없다면 default로 1로 설정
                if robot_id is None:
                    robot_id = 1
                    
                await save_robot_log(zone_id=zone_id, robot_id=robot_id, rfid_tag=rfid, plate_text=vehicle.text)
                logger.info(f"robot_logs 저장 {key}, {vehicle}")
                
        except Exception as e:
            logger.error(f"Error processing vehicle key={key}: {e}")

            
    return {"processed_count": len(vehicles)}
