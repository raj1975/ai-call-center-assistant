from unittest.mock import MagicMock, patch

import pytest

from agents.transcription_agent import _build_speaker_transcript, _normalize_numbers, _reconstruct_dialogue_with_claude, transcription_agent

_BASE_STATE = {
    "file_path": "/tmp/call.wav",
    "input_type": "audio",
    "raw_content": None,
    "metadata": None,
    "transcript": None,
    "summary": None,
    "qa_score": None,
    "errors": [],
    "routing_decision": "transcribe",
    "retry_count": 0,
    "has_sensitive_data": False,
    "sensitive_data_types": [],
    "has_profanity": False,
}


def _state(**overrides):
    return {**_BASE_STATE, **overrides}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _transcribe_response(status="COMPLETED", uri="https://s3.example.com/transcript.json", reason=None):
    job = {"TranscriptionJobStatus": status}
    if status == "COMPLETED":
        job["Transcript"] = {"TranscriptFileUri": uri}
    if reason:
        job["FailureReason"] = reason
    response = MagicMock()
    response.__getitem__ = lambda self, k: {"TranscriptionJob": job}[k]
    return {"TranscriptionJob": job}


def _transcribe_data(items=None, raw="Hello world"):
    """Minimal Transcribe JSON response."""
    return {
        "results": {
            "transcripts": [{"transcript": raw}],
            "items": items or [],
        }
    }


def _speaker_items(turns):
    """Build word-level items with speaker labels from [(speaker, text), ...]."""
    items = []
    for speaker, text in turns:
        for word in text.split():
            items.append({
                "type": "pronunciation",
                "speaker_label": speaker,
                "alternatives": [{"content": word}],
            })
    return items


# ── Routing / early-exit tests ────────────────────────────────────────────────

class TestTranscriptionAgentRouting:
    def test_error_state_skips_agent(self):
        result = transcription_agent(_state(routing_decision="error"))
        assert result == {}

    def test_non_audio_passes_through(self):
        result = transcription_agent(_state(
            input_type="transcript_txt",
            raw_content="Agent: Hi.\n\nCustomer: Hello.",
        ))
        assert result["transcript"] == "Agent: Hi.\n\nCustomer: Hello."
        assert result["routing_decision"] == "summarize"

    def test_missing_s3_bucket_returns_error(self):
        with patch.dict("os.environ", {"S3_BUCKET": ""}):
            with patch("agents.transcription_agent._S3_BUCKET", ""):
                result = transcription_agent(_state())
        assert result["routing_decision"] == "fallback"
        assert any("S3_BUCKET" in e for e in result["errors"])

    def test_unsupported_format_returns_error(self, tmp_path):
        f = tmp_path / "call.aac"
        f.write_bytes(b"\x00")
        with patch("agents.transcription_agent._S3_BUCKET", "my-bucket"):
            result = transcription_agent(_state(file_path=str(f)))
        assert result["routing_decision"] == "fallback"
        assert any("Unsupported" in e for e in result["errors"])


# ── Successful transcription + detection flags ────────────────────────────────

class TestTranscriptionAgentSuccess:
    def _run(self, transcript_text, items=None):
        """Run agent with mocked AWS and return result dict."""
        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = _transcribe_response()

        data = _transcribe_data(items=items or [], raw=transcript_text)
        mock_requests = MagicMock()
        mock_requests.get.return_value.json.return_value = data

        with patch("agents.transcription_agent._S3_BUCKET", "my-bucket"), \
             patch("boto3.client", side_effect=lambda svc, **kw: mock_s3 if svc == "s3" else mock_transcribe), \
             patch("agents.transcription_agent.requests", mock_requests), \
             patch("agents.transcription_agent._reconstruct_dialogue_with_claude", return_value=transcript_text):
            return transcription_agent(_state())

    def test_clean_transcript_no_flags(self):
        result = self._run("Agent: Hello.\n\nCustomer: Hi.")
        assert result["routing_decision"] == "summarize"
        assert result["has_sensitive_data"] is False
        assert result["has_profanity"] is False

    def test_pii_in_transcript_sets_flag(self):
        result = self._run("Agent: Can I get your SSN?\n\nCustomer: It is 523-74-1892.")
        assert result["has_sensitive_data"] is True
        assert any("Social Security" in t or "SSN" in t for t in result["sensitive_data_types"])

    def test_pci_in_transcript_sets_flag(self):
        result = self._run("Customer: My card is 4111 1111 1111 1234 expiry 09/27 CVV 342.")
        assert result["has_sensitive_data"] is True
        assert any("card" in t.lower() for t in result["sensitive_data_types"])

    def test_phi_in_transcript_sets_flag(self):
        result = self._run("Customer: My date of birth is 07/14/1978 and MRN-00445521.")
        assert result["has_sensitive_data"] is True

    def test_profanity_in_transcript_sets_flag(self):
        result = self._run("Customer: This is complete bullshit. I am so pissed off.")
        assert result["has_profanity"] is True

    def test_both_flags_set_together(self):
        result = self._run("Customer: SSN 523-74-1892. This is bullshit.")
        assert result["has_sensitive_data"] is True
        assert result["has_profanity"] is True

    def test_flags_overwrite_intake_defaults(self):
        """Transcription agent must overwrite the False flags set by intake for audio."""
        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = _transcribe_response()
        data = _transcribe_data(raw="Customer: This is bullshit and my SSN is 523-74-1892.")
        mock_requests = MagicMock()
        mock_requests.get.return_value.json.return_value = data

        with patch("agents.transcription_agent._S3_BUCKET", "my-bucket"), \
             patch("boto3.client", side_effect=lambda svc, **kw: mock_s3 if svc == "s3" else mock_transcribe), \
             patch("agents.transcription_agent.requests", mock_requests), \
             patch("agents.transcription_agent._reconstruct_dialogue_with_claude", return_value=data["results"]["transcripts"][0]["transcript"]):
            # State comes in with False (as intake sets for audio — no text to scan)
            result = transcription_agent(_state(has_sensitive_data=False, has_profanity=False))

        assert result["has_sensitive_data"] is True
        assert result["has_profanity"] is True


# ── Failure / timeout paths ───────────────────────────────────────────────────

class TestTranscriptionAgentFailures:
    def test_failed_job_returns_fallback(self):
        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = _transcribe_response(
            status="FAILED", reason="Audio too short"
        )
        with patch("agents.transcription_agent._S3_BUCKET", "my-bucket"), \
             patch("boto3.client", side_effect=lambda svc, **kw: mock_s3 if svc == "s3" else mock_transcribe):
            result = transcription_agent(_state())
        assert result["routing_decision"] == "fallback"
        assert any("failed" in e.lower() for e in result["errors"])

    def test_exception_returns_fallback(self):
        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = Exception("S3 upload failed")
        with patch("agents.transcription_agent._S3_BUCKET", "my-bucket"), \
             patch("boto3.client", side_effect=lambda svc, **kw: mock_s3 if svc == "s3" else MagicMock()):
            result = transcription_agent(_state())
        assert result["routing_decision"] == "fallback"
        assert any("Transcription error" in e for e in result["errors"])

    def test_s3_cleanup_called_on_success(self):
        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = _transcribe_response()
        data = _transcribe_data(raw="Agent: Hi.")
        mock_requests = MagicMock()
        mock_requests.get.return_value.json.return_value = data

        with patch("agents.transcription_agent._S3_BUCKET", "my-bucket"), \
             patch("boto3.client", side_effect=lambda svc, **kw: mock_s3 if svc == "s3" else mock_transcribe), \
             patch("agents.transcription_agent.requests", mock_requests), \
             patch("agents.transcription_agent._reconstruct_dialogue_with_claude", return_value="Agent: Hi."):
            transcription_agent(_state())

        mock_s3.delete_object.assert_called_once()

    def test_s3_cleanup_called_on_failure(self):
        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = _transcribe_response(status="FAILED")

        with patch("agents.transcription_agent._S3_BUCKET", "my-bucket"), \
             patch("boto3.client", side_effect=lambda svc, **kw: mock_s3 if svc == "s3" else mock_transcribe):
            transcription_agent(_state())

        mock_s3.delete_object.assert_called_once()


# ── Speaker transcript reconstruction ─────────────────────────────────────────

class TestBuildSpeakerTranscript:
    def test_speaker_labels_produce_agent_customer_turns(self):
        items = _speaker_items([
            ("spk_0", "Thank you for calling"),
            ("spk_1", "Hi I need help"),
            ("spk_0", "Of course"),
        ])
        data = _transcribe_data(items=items, raw="Thank you for calling Hi I need help Of course")
        result = _build_speaker_transcript(data)
        assert "Agent:" in result
        assert "Customer:" in result

    def test_first_speaker_is_agent(self):
        items = _speaker_items([("spk_0", "Hello welcome"), ("spk_1", "Hi there")])
        data = _transcribe_data(items=items, raw="Hello welcome Hi there")
        result = _build_speaker_transcript(data)
        assert result.startswith("Agent:")

    def test_turns_separated_by_blank_line(self):
        items = _speaker_items([("spk_0", "Hello"), ("spk_1", "Hi")])
        data = _transcribe_data(items=items, raw="Hello Hi")
        result = _build_speaker_transcript(data)
        assert "\n\n" in result

    def test_no_speaker_labels_calls_claude(self):
        data = _transcribe_data(items=[], raw="Hello how can I help you Hi I have a problem")
        with patch("agents.transcription_agent._reconstruct_dialogue_with_claude",
                   return_value="Agent: Hello how can I help you\n\nCustomer: Hi I have a problem") as mock_claude:
            result = _build_speaker_transcript(data)
        mock_claude.assert_called_once()
        assert "Agent:" in result

    def test_punctuation_attached_to_preceding_word(self):
        items = _speaker_items([("spk_0", "Hello")])
        items.append({"type": "punctuation", "alternatives": [{"content": "."}]})
        data = _transcribe_data(items=items, raw="Hello.")
        result = _build_speaker_transcript(data)
        assert "Hello." in result


class TestNormalizeNumbers:
    def test_comma_in_four_digit_number_removed(self):
        assert _normalize_numbers("visa ending in 4,421") == "visa ending in 4421"

    def test_comma_in_longer_number_removed(self):
        assert _normalize_numbers("card number 1,234,567") == "card number 1234567"

    def test_no_comma_unchanged(self):
        assert _normalize_numbers("card 4421") == "card 4421"

    def test_sentence_comma_preserved(self):
        # Comma between words, not digits — must not be removed
        assert _normalize_numbers("Hello, how are you?") == "Hello, how are you?"

    def test_mixed_text_and_number(self):
        result = _normalize_numbers("Agent: visa ending in 4,421. Customer: ok.")
        assert "4421" in result
        assert "4,421" not in result
