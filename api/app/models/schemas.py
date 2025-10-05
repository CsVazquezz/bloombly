from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date as date_type

class BloomsPredictionQuery(BaseModel):
    aoi_type: str = Field('global', description="Area of Interest type. Can be 'global', 'state', 'country', or 'bbox'.")
    date: Optional[date_type] = None
    start_date: Optional[date_type] = None
    end_date: Optional[date_type] = None
    aoi_country: Optional[str] = None
    aoi_state: Optional[str] = None
    bbox: Optional[List[float]] = None
    method: str = Field('v2', description="Prediction method. Can be 'v2', 'bloom_dynamics', 'enhanced' or 'statistical'.")

    @validator('aoi_type')
    def aoi_type_must_be_valid(cls, v):
        if v not in ['global', 'state', 'country', 'bbox']:
            raise ValueError("aoi_type must be 'global', 'state', 'country', or 'bbox'")
        return v

    @validator('method')
    def method_must_be_valid(cls, v):
        if v not in ['v2', 'bloom_dynamics', 'enhanced', 'statistical']:
            raise ValueError("method must be 'v2', 'bloom_dynamics', 'enhanced' or 'statistical'")
        return v

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values, **kwargs):
        if values.get('start_date') and v < values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v
    
    @validator('bbox', pre=True)
    def bbox_str_to_list(cls, v):
        if isinstance(v, str):
            try:
                return [float(x) for x in v.split(',')]
            except ValueError:
                raise ValueError("bbox must be a comma-separated string of four numbers")
        return v

class EnvironmentalDataQuery(BaseModel):
    lat: float = Field(..., description="Latitude of the location.")
    lon: float = Field(..., description="Longitude of the location.")
    date: date_type = Field(..., description="Date for the environmental data.")
