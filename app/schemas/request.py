from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    image_path: Optional[str] = None