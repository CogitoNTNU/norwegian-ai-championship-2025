from pydantic import BaseModel

class TumorPredictRequestDto(BaseModel):
    img: str


class TumorPredictResponseDto(BaseModel):
    img: str
