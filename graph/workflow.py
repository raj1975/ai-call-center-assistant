from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.intake_agent import intake_agent
from agents.quality_score_agent import quality_score_agent
from agents.routing_agent import (
    _error_node,
    _fallback_node,
    _route_after_intake,
    _route_after_summarization,
    _route_after_transcription,
)
from agents.summarization_agent import summarization_agent
from agents.transcription_agent import transcription_agent
from graph.state import AgentState

# In-memory checkpointer — persists graph state within a session so any run
# can be inspected or resumed by thread_id. Swap for SqliteSaver / RedisSaver
# to persist across restarts.
_checkpointer = MemorySaver()


def build_graph():
    """
    Assembles the LangGraph multi-agent pipeline.

    Flow:
        intake
          ├─(audio)──→ transcribe ──→ summarize ──→ score ──→ END
          ├─(text)───────────────────→ summarize ──→ score ──→ END
          ├─(error)──────────────────────────────────────────→ END
          └─(any failure)──→ fallback ──────────────→ score ──→ END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("intake", intake_agent)
    workflow.add_node("transcribe", transcription_agent)
    workflow.add_node("summarize", summarization_agent)
    workflow.add_node("score", quality_score_agent)
    workflow.add_node("fallback", _fallback_node)
    workflow.add_node("error_end", _error_node)

    workflow.set_entry_point("intake")

    workflow.add_conditional_edges(
        "intake",
        _route_after_intake,
        {"transcribe": "transcribe", "summarize": "summarize", "error_end": "error_end"},
    )
    workflow.add_conditional_edges(
        "transcribe",
        _route_after_transcription,
        {"summarize": "summarize", "fallback": "fallback"},
    )
    workflow.add_conditional_edges(
        "summarize",
        _route_after_summarization,
        {"score": "score", "fallback": "fallback"},
    )

    workflow.add_edge("fallback", "score")
    workflow.add_edge("score", END)
    workflow.add_edge("error_end", END)

    return workflow.compile(checkpointer=_checkpointer)


graph = build_graph()
