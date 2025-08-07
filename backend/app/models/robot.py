from typing import Dict, Optional
from pydantic import BaseModel

class VehicleInfo(BaseModel):
    text: str
    ev: Optional[str] = None  

class RobotStatusRequest(BaseModel):
    robot_id: Optional[int] = None 
    rfid: str
    vehicles: Dict[str, Optional[VehicleInfo]]  #ZONE1, ZONE2 등

class VehicleLocation(BaseModel):
    name: str
    plate_text: Optional[str]  # ← 변경
    car_type: Optional[str]    # ← 변경

class RobotVehiclesLocationResponse(BaseModel):
    robot_id: int
    rfid_tag: Optional[str]  # ✅ None 허용
    message: str
    floor: int
    vehicles : Dict[str, Optional[list[VehicleLocation]]]
