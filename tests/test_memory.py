import json

import pytest

import utils.memory as memory_module
from utils.memory import get_call, init_db, list_calls, save_call


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_call_history.db"
    monkeypatch.setattr(memory_module, "_DB_PATH", db_path)
    init_db()
    return db_path


def _state(file_name="call.json", sentiment="positive", score=8.5, agent="Alice", customer="Bob"):
    return {
        "metadata": {
            "file_name": file_name, "agent_name": agent,
            "customer_name": customer, "call_date": "2026-05-15",
        },
        "summary": {
            "overview": "Test call.", "sentiment": sentiment,
            "key_points": ["Point one"], "action_items": ["Follow up"],
            "call_outcome": "Resolved", "tags": ["billing"],
        },
        "qa_score": {"overall_score": score, "feedback": "Good."},
        "transcript": "Agent: Hi.\n\nCustomer: Hello.",
        "errors": [],
    }


class TestInitDb:
    def test_creates_db_file(self, temp_db):
        assert temp_db.exists()

    def test_idempotent_on_repeated_calls(self, temp_db, monkeypatch):
        monkeypatch.setattr(memory_module, "_DB_PATH", temp_db)
        init_db()
        init_db()
        assert temp_db.exists()


class TestSaveAndGetCall:
    def test_roundtrip(self, temp_db):
        save_call("thread_001", _state())
        row = get_call("thread_001")
        assert row is not None
        assert row["file_name"] == "call.json"
        assert row["agent_name"] == "Alice"
        assert row["customer_name"] == "Bob"
        assert row["overall_score"] == 8.5
        assert row["sentiment"] == "positive"

    def test_summary_stored_as_json_string(self, temp_db):
        save_call("thread_001", _state())
        row = get_call("thread_001")
        summary = json.loads(row["summary"])
        assert summary["overview"] == "Test call."
        assert isinstance(summary["key_points"], list)

    def test_save_replaces_existing_thread(self, temp_db):
        save_call("thread_001", _state(score=7.0))
        save_call("thread_001", _state(score=9.5))
        row = get_call("thread_001")
        assert row["overall_score"] == 9.5

    def test_get_call_returns_none_for_missing_thread(self, temp_db):
        assert get_call("nonexistent_thread") is None

    def test_transcript_stored(self, temp_db):
        save_call("thread_001", _state())
        row = get_call("thread_001")
        assert "Agent: Hi." in row["transcript"]

    def test_errors_stored_as_json(self, temp_db):
        state = _state()
        state["errors"] = ["Something went wrong"]
        save_call("thread_001", state)
        row = get_call("thread_001")
        errors = json.loads(row["errors"])
        assert errors == ["Something went wrong"]


class TestListCalls:
    def test_returns_all_saved_calls(self, temp_db):
        save_call("thread_001", _state(file_name="a.json"))
        save_call("thread_002", _state(file_name="b.json"))
        rows = list_calls()
        assert len(rows) == 2

    def test_ordered_by_created_at_desc(self, temp_db):
        save_call("thread_001", _state(file_name="first.json"))
        save_call("thread_002", _state(file_name="second.json"))
        rows = list_calls()
        assert rows[0]["file_name"] == "second.json"

    def test_limit_respected(self, temp_db):
        for i in range(5):
            save_call(f"thread_{i:03}", _state())
        rows = list_calls(limit=3)
        assert len(rows) == 3

    def test_returns_empty_list_when_db_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory_module, "_DB_PATH", tmp_path / "missing.db")
        assert list_calls() == []

    def test_returns_dicts(self, temp_db):
        save_call("thread_001", _state())
        rows = list_calls()
        assert isinstance(rows[0], dict)
