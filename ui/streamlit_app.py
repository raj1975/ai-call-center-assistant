import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env from project root before any imports that read env vars
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.workflow import graph
from utils.memory import init_db, list_calls, save_call
from utils.sensitive_data import mask_profanity, mask_sensitive

init_db()

st.set_page_config(
    page_title="AI Call Center Assistant",
    page_icon="📞",
    layout="wide",
)

st.markdown("""
<style>
  /* Hide Streamlit deploy button and top-right toolbar */
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  #MainMenu,
  header { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:6px 0 2px 0;border-left:4px solid #2563EB;padding-left:10px;margin-bottom:4px">
  <span style="font-size:1.2rem">📞</span>
  <span style="font-size:1.1rem;font-weight:700;color:#1E3A8A;letter-spacing:0.01em">Analysis Steps</span>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
<div style="display:flex;flex-direction:column;align-items:flex-start;gap:0;padding:4px 0">

  <div style="display:flex;align-items:center;gap:10px">
    <div style="background:#2563EB;color:white;border-radius:50%;width:28px;height:28px;
                display:flex;align-items:center;justify-content:center;
                font-size:0.75rem;font-weight:700;flex-shrink:0">1</div>
    <span style="font-weight:600;color:#0F172A;font-size:0.9rem">Intake</span>
  </div>

  <div style="margin-left:13px;padding:2px 0">
    <svg width="4" height="24" viewBox="0 0 4 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <line x1="2" y1="0" x2="2" y2="16" stroke="#93C5FD" stroke-width="2" stroke-dasharray="3 2"/>
      <polygon points="0,16 4,16 2,22" fill="#2563EB"/>
    </svg>
  </div>

  <div style="display:flex;align-items:center;gap:10px">
    <div style="background:#2563EB;color:white;border-radius:50%;width:28px;height:28px;
                display:flex;align-items:center;justify-content:center;
                font-size:0.75rem;font-weight:700;flex-shrink:0">2</div>
    <span style="font-weight:600;color:#0F172A;font-size:0.9rem">Transcription</span>
  </div>

  <div style="margin-left:13px;padding:2px 0">
    <svg width="4" height="24" viewBox="0 0 4 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <line x1="2" y1="0" x2="2" y2="16" stroke="#93C5FD" stroke-width="2" stroke-dasharray="3 2"/>
      <polygon points="0,16 4,16 2,22" fill="#2563EB"/>
    </svg>
  </div>

  <div style="display:flex;align-items:center;gap:10px">
    <div style="background:#2563EB;color:white;border-radius:50%;width:28px;height:28px;
                display:flex;align-items:center;justify-content:center;
                font-size:0.75rem;font-weight:700;flex-shrink:0">3</div>
    <span style="font-weight:600;color:#0F172A;font-size:0.9rem">Summarization</span>
  </div>

  <div style="margin-left:13px;padding:2px 0">
    <svg width="4" height="24" viewBox="0 0 4 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <line x1="2" y1="0" x2="2" y2="16" stroke="#93C5FD" stroke-width="2" stroke-dasharray="3 2"/>
      <polygon points="0,16 4,16 2,22" fill="#2563EB"/>
    </svg>
  </div>

  <div style="display:flex;align-items:center;gap:10px">
    <div style="background:#2563EB;color:white;border-radius:50%;width:28px;height:28px;
                display:flex;align-items:center;justify-content:center;
                font-size:0.75rem;font-weight:700;flex-shrink:0">4</div>
    <span style="font-weight:600;color:#0F172A;font-size:0.9rem">Scoring</span>
  </div>

</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:6px 0 2px 0;border-left:4px solid #0891B2;padding-left:10px;margin:12px 0 6px 0">
  <span style="font-size:1.2rem">📁</span>
  <span style="font-size:1.1rem;font-weight:700;color:#164E63;letter-spacing:0.01em">Supported Formats</span>
</div>
<div style="background:#ECFEFF;border:1px solid #A5F3FC;border-radius:10px;padding:11px 14px;margin-bottom:4px">
  <div style="display:flex;flex-direction:column;gap:6px">
    <div style="display:flex;align-items:center;gap:8px">
      <span style="background:#0891B2;color:white;border-radius:4px;padding:1px 7px;font-size:0.72rem;font-weight:600;letter-spacing:0.02em">AUDIO</span>
      <span style="color:#374151;font-size:0.82rem">mp3 &nbsp;wav &nbsp;m4a &nbsp;ogg</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px">
      <span style="background:#0E7490;color:white;border-radius:4px;padding:1px 5px;font-size:0.72rem;font-weight:600;letter-spacing:0.02em">TRANSCRIPT</span>
      <span style="color:#374151;font-size:0.82rem">json &nbsp;txt</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:6px 0 2px 0;border-left:4px solid #D97706;padding-left:10px;margin:12px 0 6px 0">
  <span style="font-size:1.2rem">🔒</span>
  <span style="font-size:1.1rem;font-weight:700;color:#78350F;letter-spacing:0.01em">Privacy &amp; Ethics</span>
</div>
<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;padding:11px 14px;margin-bottom:4px">
  <p style="color:#431407;font-size:0.8rem;line-height:1.5;margin:0 0 6px 0">
    Summaries are <strong>factual and neutral</strong> — no fabricated details or unnecessary PII exposed.
  </p>
  <p style="color:#7C2D12;font-size:0.78rem;line-height:1.4;margin:0">
    ⚠️ Avoid uploading transcripts with sensitive health, financial, or legal information.
  </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Call History ─────────────────────────────────────────────────────────
    st.markdown("**Call History**")
    history = list_calls(limit=30)
    if history:
        sentiment_icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡", "mixed": "🟠"}
        for call in history:
            icon = sentiment_icon.get(call["sentiment"], "⚪")
            label = f"{icon} {call['file_name']}  ·  {call['overall_score']:.1f}/10"
            with st.expander(label):
                st.caption(call["created_at"][:19].replace("T", " "))
                if call["agent_name"]:
                    st.write(f"**Agent:** {call['agent_name']}")
                if call["customer_name"]:
                    st.write(f"**Customer:** {call['customer_name']}")
                summary = json.loads(call["summary"]) if call["summary"] else {}
                if summary.get("overview"):
                    st.write(summary["overview"])
                if summary.get("action_items"):
                    st.markdown("**Action items:** " + " · ".join(summary["action_items"]))
    else:
        st.caption("No calls analyzed yet.")

    st.markdown("---")
    st.caption("Powered by Claude + LangGraph")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("AI Call Center Assistant")
st.caption("Upload a call recording or transcript to generate AI-powered summaries and QA scores.")

uploaded_file = st.file_uploader(
    "Upload Call Recording or Transcript",
    type=["mp3", "wav", "m4a", "ogg", "json", "txt", "pdf"],
    help="Audio files are transcribed via Amazon Transcribe. JSON/TXT transcripts are processed directly.",
)

if uploaded_file:
    col_info, col_btn = st.columns([4, 1])
    with col_info:
        st.info(f"**{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")
    with col_btn:
        analyze = st.button("Analyze Call", type="primary", use_container_width=True)

    if analyze:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            thread_id = uuid.uuid4().hex[:12]

            with st.status("Running pipeline...", expanded=True) as status:
                st.write("📥 Intake agent — validating input...")
                initial_state = {
                    "file_path": tmp_path,
                    "input_type": "",
                    "raw_content": None,
                    "metadata": None,
                    "transcript": None,
                    "summary": None,
                    "qa_score": None,
                    "errors": [],
                    "routing_decision": "",
                    "retry_count": 0,
                    "has_sensitive_data": None,
                    "sensitive_data_types": None,
                    "has_profanity": None,
                }
                result = graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": thread_id}},
                )
                status.update(label="Pipeline complete!", state="complete")

            # Persist to call history
            save_call(thread_id, result)

            if result.get("errors"):
                st.warning("Completed with warnings:\n- " + "\n- ".join(result["errors"]))

            if not result.get("summary") and not result.get("qa_score"):
                st.error("Pipeline failed. " + " | ".join(result.get("errors", ["Unknown error"])))
                st.stop()

            # ── Sensitive data / profanity banners ───────────────────────────
            if result.get("has_sensitive_data"):
                types = ", ".join(result.get("sensitive_data_types") or [])
                st.error(
                    f"⚠️ **Sensitive data detected in this transcript ({types}).** "
                    "The summary generated will not contain any PCI, PHI, or PII data."
                )
            if result.get("has_profanity"):
                st.error(
                    "⚠️ **Profanity or offensive language detected in this transcript.** "
                    "It has not been reproduced in the summary — sentiment is reflected appropriately."
                )

            # ── Results tabs ─────────────────────────────────────────────────
            tab_transcript, tab_summary, tab_qa, tab_tags = st.tabs([
                "📄 Transcript",
                "📝 Summary",
                "🎯 Quality Score",
                "🏷️ Tags & Highlights",
            ])

            # TRANSCRIPT
            with tab_transcript:
                st.subheader("Call Transcript")
                transcript = (
                    result.get("transcript")
                    or result.get("raw_content")
                    or "No transcript available."
                )
                if result.get("has_sensitive_data"):
                    transcript = mask_sensitive(transcript)
                if result.get("has_profanity"):
                    transcript = mask_profanity(transcript)
                lines_html = []
                for line in transcript.splitlines():
                    stripped = line.strip()
                    if stripped.lower().startswith("agent:"):
                        label, _, rest = stripped.partition(":")
                        lines_html.append(
                            f'<p style="color:#111111;margin:1px 0">'
                            f'<strong>{label}:</strong>{rest}</p>'
                        )
                    elif stripped.lower().startswith("customer:"):
                        label, _, rest = stripped.partition(":")
                        lines_html.append(
                            f'<p style="color:#1a56db;margin:1px 0">'
                            f'<span style="color:#1e3a8a;font-weight:600">{label}:</span>{rest}</p>'
                        )
                    elif stripped:
                        lines_html.append(
                            f'<p style="color:#444444;margin:1px 0">{stripped}</p>'
                        )
                    else:
                        lines_html.append('<div style="height:6px"></div>')
                st.markdown(
                    '<div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:8px;'
                    'padding:16px 20px;max-height:420px;overflow-y:auto;font-size:0.95rem;'
                    'line-height:1.5">' + "".join(lines_html) + "</div>",
                    unsafe_allow_html=True,
                )

            # SUMMARY
            with tab_summary:
                summary = result.get("summary")
                if summary:
                    st.subheader("Overview")
                    st.write(summary["overview"])
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Key Points")
                        for pt in summary["key_points"]:
                            st.markdown(f"- {pt}")
                    with col2:
                        st.subheader("Action Items")
                        for item in summary["action_items"]:
                            st.markdown(f"- {item}")
                    st.markdown("---")
                    sentiment_icons = {
                        "positive": "🟢", "negative": "🔴",
                        "neutral": "🟡", "mixed": "🟠",
                    }
                    icon = sentiment_icons.get(summary.get("sentiment", ""), "⚪")
                    col_s, col_o = st.columns(2)
                    col_s.metric("Sentiment", f"{icon} {summary.get('sentiment', '—').capitalize()}")
                    col_o.markdown(f"**Outcome:** {summary['call_outcome']}")
                else:
                    st.warning("Summary not available.")

            # QA SCORE
            with tab_qa:
                qa = result.get("qa_score")
                if qa:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Empathy", f"{qa['empathy_score']}/10")
                    c2.metric("Resolution", f"{qa['resolution_score']}/10")
                    c3.metric("Professionalism", f"{qa['professionalism_score']}/10")
                    c4.metric("Tone", f"{qa['tone_score']}/10")

                    overall = qa["overall_score"]
                    color = "green" if overall >= 7 else "orange" if overall >= 5 else "red"
                    st.markdown(f"### Overall Score: :{color}[{overall:.1f} / 10]")
                    st.progress(int(overall * 10))
                    st.markdown(f"**Feedback:** {qa['feedback']}")
                    st.markdown("---")
                    col_str, col_imp = st.columns(2)
                    with col_str:
                        st.subheader("Strengths")
                        for s in qa["strengths"]:
                            st.markdown(f"✅ {s}")
                    with col_imp:
                        st.subheader("Areas for Improvement")
                        for i in qa["improvements"]:
                            st.markdown(f"⚠️ {i}")
                else:
                    st.warning("QA score not available.")

            # TAGS & HIGHLIGHTS
            with tab_tags:
                summary = result.get("summary")
                if summary and summary.get("tags"):
                    st.subheader("Topic Tags")
                    tags_html = " ".join(
                        f'<span style="background:#dbeafe;color:#1e40af;padding:5px 12px;'
                        f'border-radius:999px;margin:3px;display:inline-block;font-size:0.85rem;'
                        f'font-weight:500">{tag}</span>'
                        for tag in summary["tags"]
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)
                    st.markdown("---")
                metadata = result.get("metadata")
                if metadata:
                    st.subheader("Call Metadata")
                    labels = {
                        "call_id": "Call ID",
                        "file_name": "File Name",
                        "input_type": "Input Type",
                        "duration_seconds": "Duration (s)",
                        "agent_name": "Agent",
                        "customer_name": "Customer",
                        "call_date": "Call Date",
                        "language": "Language",
                    }
                    rows = [
                        {"Field": labels.get(k, k.replace("_", " ").title()), "Value": str(v)}
                        for k, v in metadata.items()
                        if v is not None
                    ]
                    st.table(rows)


        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
