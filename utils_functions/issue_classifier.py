from pydantic import BaseModel
from typing import Optional

class IssueClassification(BaseModel):
    bike_type: Optional[str]
    part_category: Optional[str]
    part_name: Optional[str]
    position: Optional[str]
    issue: Optional[str]
    likely_service: Optional[str]