from unittest.mock import MagicMock, patch

from agents.summarization_agent import summarization_agent

_TRANSCRIPT = "Agent: Thank you for calling. How can I help?\n\nCustomer: I need help with my bill."

_BASE_STATE = {
    "file_path": None,
    "input_type": "transcript_txt",
    "raw_content": _TRANSCRIPT,
    "metadata": None,
    "transcript": _TRANSCRIPT,
    "summary": None,
    "qa_score": None,
    "errors": [],
    "routing_decision": "summarize",
    "retry_count": 0,
    "has_sensitive_data": False,
    "sensitive_data_types": [],
    "has_profanity": False,
}


def _state(**overrides):
    return {**_BASE_STATE, **overrides}


def _mock_summary():
    return MagicMock(
        sentiment="neutral",
        tags=["billing"],
        model_dump=lambda: {
            "overview": "Customer called about billing.",
            "key_points": ["billing inquiry"],
            "action_items": [],
            "sentiment": "neutral",
            "call_outcome": "resolved",
            "tags": ["billing"],
        },
    )


def _patch_chain(invoke_side_effect=None, invoke_return=None):
    """
    Returns a context-manager stack that replaces `prompt | llm | parser`
    with a controllable mock chain.
    """
    final_chain = MagicMock()
    if invoke_side_effect is not None:
        final_chain.invoke.side_effect = invoke_side_effect
    else:
        final_chain.invoke.return_value = invoke_return or _mock_summary()

    mock_prompt = MagicMock()
    mid_chain = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mid_chain)
    mid_chain.__or__ = MagicMock(return_value=final_chain)

    return mock_prompt, final_chain


class TestSummarizationAgentRouting:
    def test_error_state_skips_agent(self):
        result = summarization_agent(_state(routing_decision="error"))
        assert result == {}

    def test_short_transcript_returns_fallback(self):
        result = summarization_agent(_state(transcript="Hi."))
        assert result["routing_decision"] == "fallback"
        assert any("short" in e.lower() for e in result["errors"])

    def test_empty_transcript_returns_fallback(self):
        result = summarization_agent(_state(transcript="", raw_content=""))
        assert result["routing_decision"] == "fallback"

    def test_llm_exception_returns_fallback(self):
        mock_prompt, _ = _patch_chain(invoke_side_effect=Exception("LLM down"))

        with patch("agents.summarization_agent.ChatPromptTemplate") as mock_tpl, \
             patch("agents.summarization_agent.get_langchain_model"), \
             patch("agents.summarization_agent.PydanticOutputParser") as mock_parser:
            mock_tpl.from_messages.return_value = mock_prompt
            mock_parser.return_value.get_format_instructions.return_value = ""

            result = summarization_agent(_state())

        assert result["routing_decision"] == "fallback"
        assert any("Summarization failed" in e for e in result["errors"])


class TestSummarizationAgentSuccess:
    def test_successful_summary_routes_to_score(self):
        mock_prompt, _ = _patch_chain()

        with patch("agents.summarization_agent.ChatPromptTemplate") as mock_tpl, \
             patch("agents.summarization_agent.get_langchain_model"), \
             patch("agents.summarization_agent.PydanticOutputParser") as mock_parser:
            mock_tpl.from_messages.return_value = mock_prompt
            mock_parser.return_value.get_format_instructions.return_value = ""

            result = summarization_agent(_state())

        assert result["routing_decision"] == "score"
        assert "summary" in result

    def test_pii_flag_injects_preamble(self):
        captured = {}

        def capture(kwargs):
            captured.update(kwargs)
            return _mock_summary()

        mock_prompt, _ = _patch_chain(invoke_side_effect=capture)

        with patch("agents.summarization_agent.ChatPromptTemplate") as mock_tpl, \
             patch("agents.summarization_agent.get_langchain_model"), \
             patch("agents.summarization_agent.PydanticOutputParser") as mock_parser:
            mock_tpl.from_messages.return_value = mock_prompt
            mock_parser.return_value.get_format_instructions.return_value = ""

            summarization_agent(_state(
                has_sensitive_data=True,
                sensitive_data_types=["PII — Social Security Number"],
            ))

        assert "CRITICAL" in captured.get("preamble", "")
        assert "PII" in captured.get("preamble", "")

    def test_profanity_flag_injects_preamble(self):
        captured = {}

        def capture(kwargs):
            captured.update(kwargs)
            return _mock_summary()

        mock_prompt, _ = _patch_chain(invoke_side_effect=capture)

        with patch("agents.summarization_agent.ChatPromptTemplate") as mock_tpl, \
             patch("agents.summarization_agent.get_langchain_model"), \
             patch("agents.summarization_agent.PydanticOutputParser") as mock_parser:
            mock_tpl.from_messages.return_value = mock_prompt
            mock_parser.return_value.get_format_instructions.return_value = ""

            summarization_agent(_state(has_profanity=True))

        assert "profanity" in captured.get("preamble", "").lower()

    def test_clean_transcript_no_preamble(self):
        captured = {}

        def capture(kwargs):
            captured.update(kwargs)
            return _mock_summary()

        mock_prompt, _ = _patch_chain(invoke_side_effect=capture)

        with patch("agents.summarization_agent.ChatPromptTemplate") as mock_tpl, \
             patch("agents.summarization_agent.get_langchain_model"), \
             patch("agents.summarization_agent.PydanticOutputParser") as mock_parser:
            mock_tpl.from_messages.return_value = mock_prompt
            mock_parser.return_value.get_format_instructions.return_value = ""

            summarization_agent(_state())

        assert captured.get("preamble", "") == ""
