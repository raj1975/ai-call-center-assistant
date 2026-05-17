import json

import pytest

from agents.intake_agent import intake_agent

_BASE_STATE = {
    "file_path": "", "input_type": "", "raw_content": None,
    "metadata": None, "transcript": None, "summary": None,
    "qa_score": None, "errors": [], "routing_decision": "",
    "retry_count": 0, "has_sensitive_data": None,
    "sensitive_data_types": None, "has_profanity": None,
}


def _state(file_path: str) -> dict:
    return {**_BASE_STATE, "file_path": file_path}


class TestIntakeRouting:
    def test_missing_file_returns_error(self):
        result = intake_agent(_state("/tmp/no_such_file_xyz123.txt"))
        assert result["routing_decision"] == "error"
        assert any("not found" in e.lower() for e in result["errors"])

    def test_unsupported_extension_returns_error(self, tmp_path):
        f = tmp_path / "call.pdf"
        f.write_text("content")
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "error"
        assert any("unsupported" in e.lower() for e in result["errors"])

    def test_txt_routes_to_summarize(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("Agent: Hello.\n\nCustomer: Hi there.")
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "summarize"
        assert result["input_type"] == "transcript_txt"

    def test_json_routes_to_summarize(self, tmp_path):
        f = tmp_path / "call.json"
        f.write_text(json.dumps({"agent_name": "Alice", "transcript": "Agent: Hi.\n\nCustomer: Hello."}))
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "summarize"
        assert result["input_type"] == "transcript_json"

    def test_mp3_routes_to_transcribe(self, tmp_path):
        f = tmp_path / "call.mp3"
        f.write_bytes(b"\xff\xfb\x90\x00" * 16)
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "transcribe"
        assert result["input_type"] == "audio"

    def test_wav_routes_to_transcribe(self, tmp_path):
        f = tmp_path / "call.wav"
        f.write_bytes(b"RIFF" + b"\x00" * 40)
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "transcribe"

    def test_empty_txt_returns_error(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("   ")
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "error"

    def test_json_missing_transcript_field_returns_error(self, tmp_path):
        f = tmp_path / "call.json"
        f.write_text(json.dumps({"agent_name": "Alice"}))
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "error"
        assert any("transcript" in e.lower() for e in result["errors"])

    def test_malformed_json_returns_error(self, tmp_path):
        f = tmp_path / "call.json"
        f.write_text("{not: valid json")
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "error"
        assert any("json" in e.lower() for e in result["errors"])

    def test_json_with_empty_transcript_returns_error(self, tmp_path):
        f = tmp_path / "call.json"
        f.write_text(json.dumps({"transcript": ""}))
        result = intake_agent(_state(str(f)))
        assert result["routing_decision"] == "error"


class TestIntakeMetadata:
    def test_json_metadata_extracted(self, tmp_path):
        f = tmp_path / "call.json"
        f.write_text(json.dumps({
            "agent_name": "Nina", "customer_name": "Bob",
            "duration_seconds": 300, "transcript": "Agent: Hi.\n\nCustomer: Hello.",
        }))
        result = intake_agent(_state(str(f)))
        assert result["metadata"]["agent_name"] == "Nina"
        assert result["metadata"]["customer_name"] == "Bob"
        assert result["metadata"]["duration_seconds"] == 300

    def test_txt_metadata_has_call_id_and_filename(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("Agent: Hello.\n\nCustomer: Hi.")
        result = intake_agent(_state(str(f)))
        assert result["metadata"]["call_id"] is not None
        assert result["metadata"]["file_name"] == "call.txt"

    def test_raw_content_populated_for_txt(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("Agent: Hello.\n\nCustomer: Hi.")
        result = intake_agent(_state(str(f)))
        assert "Agent: Hello." in result["raw_content"]

    def test_raw_content_is_transcript_field_from_json(self, tmp_path):
        transcript = "Agent: Hi.\n\nCustomer: Hello."
        f = tmp_path / "call.json"
        f.write_text(json.dumps({"transcript": transcript}))
        result = intake_agent(_state(str(f)))
        assert result["raw_content"] == transcript


class TestIntakeSensitiveDataDetection:
    def test_pii_ssn_detected(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("Agent: Verify SSN.\n\nCustomer: It is 523-74-1892.")
        result = intake_agent(_state(str(f)))
        assert result["has_sensitive_data"] is True
        assert any("Social Security" in t or "SSN" in t for t in result["sensitive_data_types"])

    def test_phi_dob_detected_in_json(self, tmp_path):
        f = tmp_path / "call.json"
        f.write_text(json.dumps({"transcript": "Agent: DOB?\n\nCustomer: date of birth 07/14/1978"}))
        result = intake_agent(_state(str(f)))
        assert result["has_sensitive_data"] is True

    def test_profanity_detected(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("Agent: How can I help?\n\nCustomer: This is complete bullshit!")
        result = intake_agent(_state(str(f)))
        assert result["has_profanity"] is True

    def test_clean_transcript_no_flags(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("Agent: How can I help?\n\nCustomer: I have a billing question.")
        result = intake_agent(_state(str(f)))
        assert result["has_sensitive_data"] is False
        assert result["has_profanity"] is False

    def test_both_flags_set_together(self, tmp_path):
        f = tmp_path / "call.txt"
        f.write_text("Agent: SSN?\n\nCustomer: 523-74-1892 and this is bullshit.")
        result = intake_agent(_state(str(f)))
        assert result["has_sensitive_data"] is True
        assert result["has_profanity"] is True
