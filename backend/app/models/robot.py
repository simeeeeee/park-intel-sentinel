from typing import Dict
from pydantic import BaseModel

class VehicleInfo(BaseModel):
    text: str
    ev: str

class RobotStatusRequest(BaseModel):
    rfid: str
    vehicles: Dict[str, VehicleInfo]  # 1, 2, 3, ... 키가 string
