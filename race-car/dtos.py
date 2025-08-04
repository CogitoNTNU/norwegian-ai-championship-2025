from pydantic import BaseModel
from typing import Dict, Optional, List


class RaceCarPredictRequestDto(BaseModel):
    did_crash: bool
    elapsed_ticks: int
    distance: float
    velocity: Dict[str, float]
    sensors: Dict[str, Optional[int]]


class RaceCarPredictResponseDto(BaseModel):
    actions: List[str]
    # 'ACCELERATE'
    # 'DECELERATE'
    # 'STEER_LEFT'
    # 'STEER_RIGHT'
    # 'NOTHING''
