from typing import Dict, Optional
from pydantic import BaseModel

class VehicleInfo(BaseModel):
    text: str
    ev: str

class RobotStatusRequest(BaseModel):
    robot_id: Optional[int] = None  # 있어도 되고 없어도 됨
    rfid: str
    vehicles: Dict[str, VehicleInfo]  # 1, 2, 3, ... 키가 string
