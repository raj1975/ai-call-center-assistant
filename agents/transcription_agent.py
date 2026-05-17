import json
import os
import re
import time
import uuid
from pathlib import Path

import requests

import boto3

from graph.state import AgentState
from utils.observability import log_agent
from utils.sensitive_data import detect_profanity, detect_sensitive

_REGION = os.getenv("AWS_REGION", "us-east-1")
_S3_BUCKET = os.getenv("S3_BUCKET", "")

# Amazon Transcribe accepted format strings per file extension
_FORMAT_MAP = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".flac": "flac",
    ".ogg": "ogg",
    ".webm": "webm",
    ".m4a": "mp4",   # m4a is an mp4 container
    ".mp4": "mp4",
    ".amr": "amr",
}

_POLL_INTERVAL = 5    # seconds between status checks
_TIMEOUT = 300        # give up after 5 minutes


def _normalize_numbers(text: str) -> str:
    """Remove comma thousand-separators that Amazon Transcribe inserts (e.g. '4,421' → '4421')."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'(\d),(\d{3})\b', r'\1\2', text)
    return text


def _build_speaker_transcript(data: dict) -> str:
    """
    Reconstructs a dialogue transcript from Transcribe speaker-diarized output.
    spk_0 (first speaker) → Agent, spk_1 → Customer.
    Falls back to Claude reconstruction when speaker labels are absent.
    """
    items = data.get("results", {}).get("items", [])
    raw = data["results"]["transcripts"][0]["transcript"]

    if not any(i.get("speaker_label") for i in items):
        return _reconstruct_dialogue_with_claude(raw)

    # Map speaker IDs to roles — first speaker encountered is the agent
    role_map: dict[str, str] = {}
    segments: list[tuple[str, str]] = []  # (role, word)

    for item in items:
        if item.get("type") != "pronunciation":
            if segments and item.get("alternatives"):
                last_role, last_word = segments[-1]
                segments[-1] = (last_role, last_word + item["alternatives"][0]["content"])
            continue

        speaker = item.get("speaker_label", "spk_0")
        if speaker not in role_map:
            role_map[speaker] = "Agent" if len(role_map) == 0 else "Customer"

        role = role_map[speaker]
        word = item["alternatives"][0]["content"]
        segments.append((role, word))

    # Group consecutive words by speaker into turns
    current_role = None
    current_words: list[str] = []
    lines: list[str] = []

    for role, word in segments:
        if role != current_role:
            if current_role and current_words:
                lines.append(f"{current_role}: {' '.join(current_words)}")
            current_role = role
            current_words = [word]
        else:
            current_words.append(word)

    if current_role and current_words:
        lines.append(f"{current_role}: {' '.join(current_words)}")

    return "\n\n".join(lines)


def _reconstruct_dialogue_with_claude(raw: str) -> str:
    """
    Uses Claude to split a flat transcript into Agent / Customer turns.
    Called when Amazon Transcribe returns no speaker labels.
    """
    from utils.llm_factory import get_anthropic_client, get_model_id

    client = get_anthropic_client()
    response = client.messages.create(
        model=get_model_id(),
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": (
                "The following text is a call center conversation that was transcribed from audio "
                "without speaker labels. Reconstruct it as a properly labelled dialogue.\n\n"
                "Rules:\n"
                "- Label each turn as either 'Agent:' or 'Customer:'\n"
                "- The agent speaks first (answers the call), uses professional language, "
                "and guides the conversation toward resolution\n"
                "- The customer has a question, issue, or complaint\n"
                "- Separate each turn with a blank line\n"
                "- Do NOT add, remove, or paraphrase any words — only add speaker labels\n\n"
                f"Transcript:\n{raw}"
            ),
        }],
    )
    return response.content[0].text.strip()


@log_agent("transcription_agent")
def transcription_agent(state: AgentState) -> dict:
    """Converts audio files to text using Amazon Transcribe. Falls back gracefully on failure."""
    if state.get("routing_decision") == "error":
        return {}

    # Non-audio inputs already have raw_content — detection already ran in intake
    if state["input_type"] != "audio":
        return {"transcript": state.get("raw_content", ""), "routing_decision": "summarize"}

    if not _S3_BUCKET:
        return {
            "errors": ["S3_BUCKET env var not set — required for Amazon Transcribe"],
            "routing_decision": "fallback",
        }

    suffix = Path(state["file_path"]).suffix.lower()
    media_format = _FORMAT_MAP.get(suffix)
    if not media_format:
        return {
            "errors": [f"Unsupported audio format for Amazon Transcribe: {suffix}"],
            "routing_decision": "fallback",
        }

    s3 = boto3.client("s3", region_name=_REGION)
    transcribe = boto3.client("transcribe", region_name=_REGION)

    s3_key = f"call-audio/{uuid.uuid4().hex}{suffix}"
    job_name = f"call-{uuid.uuid4().hex[:12]}"

    try:
        s3.upload_file(state["file_path"], _S3_BUCKET, s3_key)

        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": f"s3://{_S3_BUCKET}/{s3_key}"},
            MediaFormat=media_format,
            LanguageCode="en-US",
            Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": 2},
        )

        deadline = time.time() + _TIMEOUT
        while time.time() < deadline:
            response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                data = requests.get(uri, timeout=30).json()
                transcript = _normalize_numbers(_build_speaker_transcript(data))
                sensitive = detect_sensitive(transcript)
                return {
                    "transcript": transcript,
                    "routing_decision": "summarize",
                    "has_sensitive_data": sensitive["has_sensitive_data"],
                    "sensitive_data_types": sensitive["sensitive_data_types"],
                    "has_profanity": detect_profanity(transcript),
                }

            if status == "FAILED":
                reason = response["TranscriptionJob"].get("FailureReason", "unknown")
                return {"errors": [f"Transcription job failed: {reason}"], "routing_decision": "fallback"}

            time.sleep(_POLL_INTERVAL)

        return {"errors": ["Transcription timed out after 5 minutes"], "routing_decision": "fallback"}

    except Exception as e:
        return {"errors": [f"Transcription error: {e}"], "routing_decision": "fallback"}

    finally:
        try:
            s3.delete_object(Bucket=_S3_BUCKET, Key=s3_key)
        except Exception:
            pass
