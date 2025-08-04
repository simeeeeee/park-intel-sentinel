from typing import Dict, Optional
from pydantic import BaseModel

class VehicleInfo(BaseModel):
    text: str
    ev: Optional[str] = None  

class RobotStatusRequest(BaseModel):
    robot_id: Optional[int] = None 
    rfid: str
    vehicles: Dict[str, Optional[VehicleInfo]]  #ZONE1, ZONE2 등