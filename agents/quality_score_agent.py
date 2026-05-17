from pydantic import ValidationError

from graph.state import AgentState
from utils.llm_factory import get_anthropic_client, get_model_id
from utils.observability import log_agent
from utils.validation import QAScore

# Rubric cached as ephemeral system prompt — reused across scoring calls
_RUBRIC = """You are a QA analyst for a call center. Score the agent's performance on four dimensions (0–10 each):

- empathy_score: Did the agent show genuine understanding and compassion toward the customer?
- resolution_score: Was the customer's issue resolved effectively and efficiently?
- professionalism_score: Was the agent's language, demeanor, and conduct professional throughout?
- tone_score: Was the agent's tone warm, positive, and appropriate at all times?

Compute: overall_score = (empathy * 0.25) + (resolution * 0.35) + (professionalism * 0.20) + (tone * 0.20)

Provide:
- feedback: 2-3 sentence overall assessment
- strengths: list of specific positive behaviors observed
- improvements: list of specific areas that need improvement

Base all scores strictly on evidence from the transcript."""

_TOOL = {
    "name": "submit_qa_score",
    "description": "Submit the structured QA evaluation for the call",
    "input_schema": {
        "type": "object",
        "properties": {
            "empathy_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "resolution_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "professionalism_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "tone_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "overall_score": {"type": "number"},
            "feedback": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "improvements": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "empathy_score", "resolution_score", "professionalism_score",
            "tone_score", "overall_score", "feedback", "strengths", "improvements",
        ],
    },
}


def _fallback_score() -> dict:
    return QAScore(
        empathy_score=5,
        resolution_score=5,
        professionalism_score=5,
        tone_score=5,
        overall_score=5.0,
        feedback="Automated scoring unavailable. Manual review is recommended.",
        strengths=["Unable to assess automatically"],
        improvements=["Schedule manual QA review"],
    ).model_dump()


@log_agent("quality_score_agent")
def quality_score_agent(state: AgentState) -> dict:
    """Scores call quality via Claude function calling with a structured rubric."""
    if state.get("routing_decision") == "error":
        return {}

    transcript = state.get("transcript") or state.get("raw_content", "")
    if not transcript:
        return {"qa_score": _fallback_score(), "routing_decision": "complete"}

    client = get_anthropic_client()
    try:
        response = client.messages.create(
            model=get_model_id(),
            max_tokens=1024,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "submit_qa_score"},
            system=[{"type": "text", "text": _RUBRIC, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": f"Score this call transcript:\n\n{transcript[:4000]}",
            }],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_qa_score":
                raw = dict(block.input)
                for field in ("strengths", "improvements"):
                    val = raw.get(field)
                    if isinstance(val, str):
                        items = [
                            line.lstrip("•-* ").strip()
                            for line in val.splitlines()
                            if line.strip().lstrip("•-* ").strip()
                        ]
                        raw[field] = items if items else [val.strip()]
                score = QAScore(**raw)
                return {"qa_score": score.model_dump(), "routing_decision": "complete"}
    except ValidationError as e:
        return {"errors": [f"QA score validation failed: {e}"], "qa_score": _fallback_score(), "routing_decision": "complete"}
    except Exception as e:
        return {"errors": [f"QA scoring failed: {e}"], "qa_score": _fallback_score(), "routing_decision": "complete"}

    return {"qa_score": _fallback_score(), "routing_decision": "complete"}
