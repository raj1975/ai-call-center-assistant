from graph.state import AgentState


def _route_after_intake(state: AgentState) -> str:
    decision = state.get("routing_decision", "error")
    if decision == "error":
        return "error_end"
    return "transcribe" if decision == "transcribe" else "summarize"


def _route_after_transcription(state: AgentState) -> str:
    return "fallback" if state.get("routing_decision") == "fallback" else "summarize"


def _route_after_summarization(state: AgentState) -> str:
    return "fallback" if state.get("routing_decision") == "fallback" else "score"


def _fallback_node(state: AgentState) -> dict:
    """Continues pipeline with partial data rather than terminating."""
    return {"routing_decision": "score"}


def _error_node(state: AgentState) -> dict:
    """Terminal node for unrecoverable errors (bad file, unsupported format)."""
    return {}
