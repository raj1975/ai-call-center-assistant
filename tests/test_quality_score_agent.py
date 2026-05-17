from unittest.mock import MagicMock, patch

import pytest

from agents.quality_score_agent import quality_score_agent

_BASE_STATE = {
    "file_path": "", "input_type": "transcript_txt",
    "raw_content": "Agent: Hi.\n\nCustomer: Hello.",
    "metadata": None, "transcript": "Agent: Hi.\n\nCustomer: Hello.",
    "summary": None, "qa_score": None, "errors": [],
    "routing_decision": "score", "retry_count": 0,
    "has_sensitive_data": False, "sensitive_data_types": [], "has_profanity": False,
}


def _state(**overrides):
    return {**_BASE_STATE, **overrides}


def _mock_response(strengths, improvements):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_qa_score"
    block.input = {
        "empathy_score": 8, "resolution_score": 7,
        "professionalism_score": 9, "tone_score": 8,
        "overall_score": 7.95, "feedback": "Good call overall.",
        "strengths": strengths, "improvements": improvements,
    }
    response = MagicMock()
    response.content = [block]
    return response


class TestQualityScoreAgent:
    def test_valid_list_response_passes_through(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response(
            strengths=["Showed empathy", "Resolved quickly"],
            improvements=["Could be warmer"],
        )
        with patch("agents.quality_score_agent.get_anthropic_client", return_value=mock_client):
            result = quality_score_agent(_state())
        assert result["routing_decision"] == "complete"
        assert result["qa_score"]["empathy_score"] == 8
        assert isinstance(result["qa_score"]["strengths"], list)
        assert isinstance(result["qa_score"]["improvements"], list)

    def test_string_strengths_coerced_to_list(self):
        """Regression: LLM sometimes returns bullet-point string instead of JSON array."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response(
            strengths="\n- Showed empathy\n- Resolved the issue quickly\n",
            improvements="\n- Could acknowledge wait time earlier\n- Offer a callback option\n",
        )
        with patch("agents.quality_score_agent.get_anthropic_client", return_value=mock_client):
            result = quality_score_agent(_state())
        assert isinstance(result["qa_score"]["strengths"], list)
        assert len(result["qa_score"]["strengths"]) == 2
        assert result["qa_score"]["strengths"][0] == "Showed empathy"
        assert isinstance(result["qa_score"]["improvements"], list)
        assert len(result["qa_score"]["improvements"]) == 2

    def test_bullet_with_asterisks_coerced(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response(
            strengths="* Great tone\n* Clear communication\n",
            improvements="* Needs faster resolution\n",
        )
        with patch("agents.quality_score_agent.get_anthropic_client", return_value=mock_client):
            result = quality_score_agent(_state())
        assert result["qa_score"]["strengths"] == ["Great tone", "Clear communication"]
        assert result["qa_score"]["improvements"] == ["Needs faster resolution"]

    def test_error_routing_skips_agent(self):
        result = quality_score_agent(_state(routing_decision="error"))
        assert result == {}

    def test_missing_transcript_returns_fallback(self):
        with patch("agents.quality_score_agent.get_anthropic_client"):
            result = quality_score_agent(_state(transcript=None, raw_content=None))
        assert result["qa_score"] is not None
        assert result["routing_decision"] == "complete"

    def test_api_exception_returns_fallback_with_error(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Connection error")
        with patch("agents.quality_score_agent.get_anthropic_client", return_value=mock_client):
            result = quality_score_agent(_state())
        assert result["qa_score"]["feedback"] is not None
        assert any("QA scoring failed" in e for e in result["errors"])
        assert result["routing_decision"] == "complete"

    def test_fallback_score_has_required_fields(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("fail")
        with patch("agents.quality_score_agent.get_anthropic_client", return_value=mock_client):
            result = quality_score_agent(_state())
        qa = result["qa_score"]
        for field in ("empathy_score", "resolution_score", "professionalism_score",
                      "tone_score", "overall_score", "feedback", "strengths", "improvements"):
            assert field in qa
