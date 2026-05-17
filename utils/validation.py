from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CallMetadata(BaseModel):
    call_id: str
    file_name: str
    input_type: Literal["audio", "transcript_json", "transcript_txt"]
    duration_seconds: Optional[float] = None
    agent_name: Optional[str] = None
    customer_name: Optional[str] = None
    call_date: Optional[str] = None
    language: str = "en"


class CallSummary(BaseModel):
    overview: str = Field(description="2-3 sentence overview of the call")
    key_points: List[str] = Field(description="Main discussion points")
    action_items: List[str] = Field(description="Follow-up actions required")
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    call_outcome: str = Field(description="Resolution status")
    tags: List[str] = Field(description="Topic tags for categorization")


class QAScore(BaseModel):
    empathy_score: int = Field(ge=0, le=10, description="Agent empathy and compassion")
    resolution_score: int = Field(ge=0, le=10, description="Effectiveness of issue resolution")
    professionalism_score: int = Field(ge=0, le=10, description="Language and conduct quality")
    tone_score: int = Field(ge=0, le=10, description="Positivity and appropriateness of tone")
    overall_score: float = Field(description="Weighted average score")
    feedback: str = Field(description="Overall assessment feedback")
    strengths: List[str]
    improvements: List[str]
