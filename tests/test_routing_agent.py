from agents.routing_agent import (
    _error_node,
    _fallback_node,
    _route_after_intake,
    _route_after_summarization,
    _route_after_transcription,
)


class TestRouteAfterIntake:
    def test_error_routes_to_error_end(self):
        assert _route_after_intake({"routing_decision": "error"}) == "error_end"

    def test_transcribe_routes_to_transcribe(self):
        assert _route_after_intake({"routing_decision": "transcribe"}) == "transcribe"

    def test_summarize_routes_to_summarize(self):
        assert _route_after_intake({"routing_decision": "summarize"}) == "summarize"

    def test_missing_decision_defaults_to_error(self):
        assert _route_after_intake({}) == "error_end"

    def test_empty_string_routes_to_summarize(self):
        # Any non-error, non-transcribe value falls through to summarize
        assert _route_after_intake({"routing_decision": ""}) == "summarize"


class TestRouteAfterTranscription:
    def test_fallback_routes_to_fallback(self):
        assert _route_after_transcription({"routing_decision": "fallback"}) == "fallback"

    def test_success_routes_to_summarize(self):
        assert _route_after_transcription({"routing_decision": "summarize"}) == "summarize"

    def test_any_non_fallback_routes_to_summarize(self):
        assert _route_after_transcription({"routing_decision": "score"}) == "summarize"


class TestRouteAfterSummarization:
    def test_fallback_routes_to_fallback(self):
        assert _route_after_summarization({"routing_decision": "fallback"}) == "fallback"

    def test_score_routes_to_score(self):
        assert _route_after_summarization({"routing_decision": "score"}) == "score"

    def test_any_non_fallback_routes_to_score(self):
        assert _route_after_summarization({"routing_decision": "complete"}) == "score"


class TestFallbackNode:
    def test_sets_routing_to_score(self):
        result = _fallback_node({"routing_decision": "fallback", "errors": ["oops"]})
        assert result == {"routing_decision": "score"}

    def test_does_not_return_other_keys(self):
        result = _fallback_node({"routing_decision": "fallback"})
        assert list(result.keys()) == ["routing_decision"]


class TestErrorNode:
    def test_returns_empty_dict(self):
        assert _error_node({"routing_decision": "error"}) == {}

    def test_state_not_mutated(self):
        state = {"routing_decision": "error", "errors": ["bad file"]}
        _error_node(state)
        assert state["errors"] == ["bad file"]
