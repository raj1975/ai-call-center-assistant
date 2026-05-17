import json
import uuid
from datetime import datetime
from pathlib import Path

from graph.state import AgentState
from utils.observability import log_agent
from utils.sensitive_data import detect_profanity, detect_sensitive
from utils.validation import CallMetadata

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
ALL_VALID_EXTENSIONS = AUDIO_EXTENSIONS | {".json", ".txt", ".text"}


@log_agent("intake_agent")
def intake_agent(state: AgentState) -> dict:
    """Validates input format, detects type, and extracts call metadata."""
    path = Path(state["file_path"])

    if not path.exists():
        return {"errors": [f"File not found: {state['file_path']}"], "routing_decision": "error"}

    suffix = path.suffix.lower()
    if suffix not in ALL_VALID_EXTENSIONS:
        return {"errors": [f"Unsupported file format '{suffix}'. Supported: audio (mp3/wav/m4a/ogg/flac) or transcript (json/txt)"], "routing_decision": "error"}

    if suffix in AUDIO_EXTENSIONS:
        input_type = "audio"
    elif suffix == ".json":
        input_type = "transcript_json"
    else:
        input_type = "transcript_txt"

    metadata = CallMetadata(
        call_id=str(uuid.uuid4())[:8].upper(),
        file_name=path.name,
        input_type=input_type,
        call_date=datetime.now().isoformat(),
    )

    raw_content = None

    if input_type == "transcript_json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata.agent_name = data.get("agent_name")
            metadata.customer_name = data.get("customer_name")
            metadata.duration_seconds = data.get("duration_seconds")
            raw_content = data.get("transcript", "")
            if not raw_content:
                return {"errors": ["JSON file missing 'transcript' field"], "routing_decision": "error"}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {"errors": [f"Invalid JSON: {e}"], "routing_decision": "error"}
    elif input_type == "transcript_txt":
        raw_content = path.read_text(encoding="utf-8")
        if not raw_content.strip():
            return {"errors": ["Transcript file is empty"], "routing_decision": "error"}

    routing = "transcribe" if input_type == "audio" else "summarize"

    # Run sensitive data and profanity detection on available text
    text_to_scan = raw_content or ""
    sensitive = detect_sensitive(text_to_scan)
    profanity = detect_profanity(text_to_scan)

    return {
        "input_type": input_type,
        "raw_content": raw_content,
        "metadata": metadata.model_dump(),
        "routing_decision": routing,
        "has_sensitive_data": sensitive["has_sensitive_data"],
        "sensitive_data_types": sensitive["sensitive_data_types"],
        "has_profanity": profanity,
    }
