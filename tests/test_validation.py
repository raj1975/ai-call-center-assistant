import pytest
from pydantic import ValidationError

from utils.validation import CallMetadata, CallSummary, QAScore


class TestCallMetadata:
    def test_valid_json_transcript(self):
        m = CallMetadata(call_id="ABC123", file_name="call.json", input_type="transcript_json")
        assert m.language == "en"
        assert m.duration_seconds is None
        assert m.agent_name is None

    def test_valid_txt_transcript(self):
        m = CallMetadata(call_id="XYZ", file_name="call.txt", input_type="transcript_txt")
        assert m.input_type == "transcript_txt"

    def test_valid_audio(self):
        m = CallMetadata(call_id="XYZ", file_name="call.mp3", input_type="audio")
        assert m.input_type == "audio"

    def test_invalid_input_type_raises(self):
        with pytest.raises(ValidationError):
            CallMetadata(call_id="X", file_name="f.txt", input_type="video")

    def test_optional_fields_populated(self):
        m = CallMetadata(
            call_id="X", file_name="f.json", input_type="transcript_json",
            agent_name="Alice", customer_name="Bob", duration_seconds=300,
        )
        assert m.agent_name == "Alice"
        assert m.customer_name == "Bob"
        assert m.duration_seconds == 300

    def test_model_dump_returns_dict(self):
        m = CallMetadata(call_id="X", file_name="f.json", input_type="transcript_json")
        d = m.model_dump()
        assert isinstance(d, dict)
        assert d["call_id"] == "X"


class TestCallSummary:
    def _make(self, **overrides):
        base = dict(
            overview="A customer called about a billing issue.",
            key_points=["Customer disputed a charge"],
            action_items=["Issue refund within 3 days"],
            sentiment="negative",
            call_outcome="Resolved — refund issued",
            tags=["billing", "refund"],
        )
        base.update(overrides)
        return CallSummary(**base)

    def test_valid_summary(self):
        s = self._make()
        assert s.sentiment == "negative"
        assert len(s.key_points) == 1

    def test_all_valid_sentiments(self):
        for sentiment in ("positive", "neutral", "negative", "mixed"):
            s = self._make(sentiment=sentiment)
            assert s.sentiment == sentiment

    def test_invalid_sentiment_raises(self):
        with pytest.raises(ValidationError):
            self._make(sentiment="angry")

    def test_tags_is_list(self):
        s = self._make()
        assert isinstance(s.tags, list)

    def test_model_dump_serializes_correctly(self):
        s = self._make()
        d = s.model_dump()
        assert d["overview"] == "A customer called about a billing issue."
        assert isinstance(d["key_points"], list)
        assert isinstance(d["action_items"], list)


class TestQAScore:
    def _make(self, **overrides):
        base = dict(
            empathy_score=8, resolution_score=9,
            professionalism_score=7, tone_score=8,
            overall_score=8.15, feedback="Good call overall.",
            strengths=["Showed genuine empathy", "Resolved issue efficiently"],
            improvements=["Could have been warmer"],
        )
        base.update(overrides)
        return QAScore(**base)

    def test_valid_score(self):
        q = self._make()
        assert q.overall_score == 8.15
        assert q.empathy_score == 8

    def test_score_above_max_raises(self):
        with pytest.raises(ValidationError):
            self._make(empathy_score=11)

    def test_score_below_min_raises(self):
        with pytest.raises(ValidationError):
            self._make(resolution_score=-1)

    def test_score_boundary_values(self):
        q = self._make(empathy_score=0, tone_score=10)
        assert q.empathy_score == 0
        assert q.tone_score == 10

    def test_strengths_is_list(self):
        q = self._make()
        assert isinstance(q.strengths, list)

    def test_improvements_is_list(self):
        q = self._make()
        assert isinstance(q.improvements, list)

    def test_model_dump_contains_all_fields(self):
        q = self._make()
        d = q.model_dump()
        for key in ("empathy_score", "resolution_score", "professionalism_score",
                    "tone_score", "overall_score", "feedback", "strengths", "improvements"):
            assert key in d
