from typing import Any, List, Optional
import datetime
from pydantic import BaseModel

class DataInputSchema(BaseModel):
    inputs: Optional[str] 
    
class PredictionResults(BaseModel):
    #errors: Optional[Any]
    #version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]

