import operator
from typing import Annotated, List, Optional, TypedDict


class AgentState(TypedDict):
    file_path: str
    input_type: str
    raw_content: Optional[str]
    metadata: Optional[dict]
    transcript: Optional[str]
    summary: Optional[dict]
    qa_score: Optional[dict]
    errors: Annotated[List[str], operator.add]  # accumulated across nodes, never overwritten
    routing_decision: str
    retry_count: int
    has_sensitive_data: Optional[bool]
    sensitive_data_types: Optional[List[str]]
    has_profanity: Optional[bool]
