from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState
from utils.llm_factory import get_langchain_model
from utils.observability import get_logger, log_agent
from utils.validation import CallSummary

logger = get_logger("summarization_agent")

# Ethics: prompt instructs the model to stay factual, neutral, and privacy-aware
_SYSTEM = """You are an expert call center analyst. Analyze call transcripts and produce structured summaries.

Rules:
- Be factual and neutral — only report what is present in the transcript.
- Do not fabricate, infer, or embellish details.
- Do not expose unnecessary personally identifiable information (PII); refer to participants by role (Agent/Customer) unless a name is directly relevant to the outcome.
- Flag sensitive topics (financial data, health info, legal disputes) in the tags field.

{format_instructions}"""


@log_agent("summarization_agent")
def summarization_agent(state: AgentState) -> dict:
    """Generates a structured call summary using LangChain + Pydantic (Anthropic or Bedrock)."""
    if state.get("routing_decision") == "error":
        return {}

    transcript = state.get("transcript") or state.get("raw_content", "")
    if not transcript or len(transcript.strip()) < 20:
        logger.warning("Transcript too short to summarize")
        return {"errors": ["Transcript too short to summarize"], "routing_decision": "fallback"}

    parser = PydanticOutputParser(pydantic_object=CallSummary)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "{preamble}Summarize this call transcript:\n\n{transcript}"),
    ])
    chain = prompt | get_langchain_model(max_tokens=1024) | parser

    # Build per-call preamble for sensitive / profanity flags
    preamble_lines = []
    if state.get("has_sensitive_data"):
        types = ", ".join(state.get("sensitive_data_types") or [])
        preamble_lines.append(
            f"CRITICAL: This transcript contains sensitive data ({types}). "
            "You MUST NOT include any PCI, PHI, or PII data — no card numbers, SSNs, "
            "emails, phone numbers, account numbers, medical records, or dates of birth — "
            "anywhere in your output."
        )
    if state.get("has_profanity"):
        preamble_lines.append(
            "NOTE: This transcript contains profanity or offensive language. "
            "Do NOT reproduce any such language in the summary. "
            "Reflect the emotional tone and sentiment accurately without quoting the words."
        )
    preamble = ("\n".join(preamble_lines) + "\n\n") if preamble_lines else ""

    try:
        summary: CallSummary = chain.invoke({
            "transcript": transcript[:6000],
            "format_instructions": parser.get_format_instructions(),
            "preamble": preamble,
        })
        logger.info("summary produced | sentiment=%s | tags=%s", summary.sentiment, summary.tags)
        return {"summary": summary.model_dump(), "routing_decision": "score"}
    except Exception as e:
        logger.error("Summarization failed: %s", e)
        return {"errors": [f"Summarization failed: {e}"], "routing_decision": "fallback"}
