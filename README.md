# AI Call Center Assistant

A modular multi-agent system that converts raw call recordings and transcripts into structured summaries and QA scores using LangGraph and Claude (Anthropic API or Amazon Bedrock).

## Architecture

```mermaid
flowchart TD
    A([File Upload<br/>Audio / JSON / TXT]) --> B

    subgraph PIPELINE["LangGraph Pipeline"]
        direction TB

        B["Intake Agent<br/>Validate format · Extract metadata<br/>Detect PII / PHI / PCI · Detect profanity"]

        B --> R1{routing_decision}

        R1 -- audio --> C["Transcription Agent<br/>Upload to S3 · Start Transcribe job<br/>Speaker diarization · Download · Cleanup"]

        C --> D

        R1 -- summarize --> D["Summarization Agent<br/>Claude · Structured output<br/>Overview · Key points · Action items<br/>Sentiment · Call outcome · Tags"]

        D --> R2{routing_decision}

        R2 -- score --> E["QA Scoring Agent<br/>Claude function calling<br/>Empathy · Resolution<br/>Professionalism · Tone"]

        R2 -- fallback --> F["Fallback Node<br/>Preserve partial state<br/>Route to scoring"]

        F --> E
    end

    E --> G[("SQLite<br/>Call History")]
    E --> S

    subgraph STREAMLIT["Streamlit UI"]
        direction TB

        S["Read AgentState"]

        S -- has_sensitive_data --> B1["Red Banner<br/>PII / PHI / PCI detected<br/>Values masked with ####"]
        S -- has_profanity --> B2["Red Banner<br/>Profanity detected<br/>Words masked with ####"]

        S --> T1["Transcript Tab<br/>Color-coded dialogue<br/>Agent: black bold · Customer: blue<br/>Sensitive values masked"]

        S --> T2["Summary Tab<br/>Overview · Key points<br/>Action items · Sentiment · Outcome"]

        S --> T3["QA Score Tab<br/>Empathy · Resolution<br/>Professionalism · Tone<br/>Overall score · Feedback"]

        S --> T4["Tags and Highlights Tab<br/>Topic tags · Call metadata table"]
    end
```

Conditional edges in the LangGraph pipeline handle routing after each node. A transcription or summarization error routes through a fallback node so QA scoring always runs on available data.

---

## Features

- **Structured summaries** — overview, key points, action items, call outcome, sentiment
- **QA scoring** — empathy, resolution, professionalism, tone scored 0–10 with feedback
- **Sensitive data detection** — PCI (card numbers, CVV), PII (SSN, email, phone), PHI (MRN, DOB, medical info) detected at intake and after audio transcription; masked in the transcript view with partial masking where applicable (SSN shows last 4 digits as `###-##-1234`; DOB shows year only as `##/##/YYYY` or `[DOB: YYYY]`; all other values replaced with `####`); suppressed entirely from the summary; red warning banner shown in UI
- **Profanity detection** — offensive language detected, masked with `####` in the transcript view, and suppressed from the summary; sentiment still reflected accurately; red warning banner shown in UI
- **Call history** — all analyzed calls persisted in SQLite and shown in the sidebar with sentiment indicator and score
- **Dual LLM mode** — `USE_BEDROCK=false` uses Anthropic API directly (local dev); `USE_BEDROCK=true` uses Claude on Amazon Bedrock (AWS deployment)
- **Audio transcription** — Amazon Transcribe via S3 upload (no OpenAI dependency)

---

## Local Setup

### Prerequisites

- Python 3.11+
- An Anthropic API key (`ANTHROPIC_API_KEY`) **or** AWS credentials with Amazon Bedrock access
- AWS credentials only needed for audio file transcription (Amazon Transcribe + S3) or Bedrock mode

### Steps

**1. Clone the repo**

```bash
git clone <repo-url>
cd ai-call-center-summarization
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure credentials**

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
# Use Anthropic API directly for local dev
USE_BEDROCK=false
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-sonnet-4-5

# Required only for audio file uploads (Amazon Transcribe uses S3)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
S3_BUCKET=your-s3-bucket-name
```

**5. Run the app**

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploy to AWS — EC2 with Docker

### Prerequisites

- An EC2 instance (recommended: `t3.medium` or larger, Amazon Linux 2023 or Ubuntu 22.04)
- Docker installed on the instance
- An IAM role attached to the EC2 instance with Bedrock, S3, and Transcribe permissions (see below)
- Port 8501 open in the instance's security group

### Step 1 — Launch EC2 and attach an IAM role

Create an IAM role with the following inline policy and attach it to your EC2 instance:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::your-s3-bucket-name/call-audio/*"
    },
    {
      "Effect": "Allow",
      "Action": ["transcribe:StartTranscriptionJob", "transcribe:GetTranscriptionJob"],
      "Resource": "*"
    }
  ]
}
```

> With an instance role attached, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are not needed — the AWS SDK picks up credentials automatically from the instance metadata service.

### Step 2 — Install Docker on the EC2 instance

```bash
# Amazon Linux 2023
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user   # allow running docker without sudo

# Ubuntu 22.04
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

Log out and back in for the group change to take effect.

### Step 3 — Copy the project to EC2

```bash
# From your local machine
scp -r ./ ec2-user@<ec2-public-ip>:~/ai-call-center-summarization
ssh ec2-user@<ec2-public-ip>
cd ai-call-center-summarization
```

Or clone directly on the instance if the repo is on GitHub:

```bash
git clone <repo-url>
cd ai-call-center-summarization
```

### Step 4 — Create the .env file on EC2

```bash
cp .env.example .env
nano .env
```

Set at minimum:

```
USE_BEDROCK=true
AWS_REGION=us-east-1
BEDROCK_PRIMARY_MODEL=us.anthropic.claude-sonnet-4-5-20250514-v1:0
S3_BUCKET=your-s3-bucket-name
```

Leave `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` blank — the instance role handles auth.

### Step 5 — Build and run with Docker

```bash
docker compose up --build -d
```

The app will be available at:

```
http://<ec2-public-ip>:8501
```

### Step 6 — Open port 8501 in the security group

In the AWS Console go to **EC2 → Security Groups → Inbound rules** and add:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| Custom TCP | TCP | 8501 | 0.0.0.0/0 (or your IP) |

### Useful commands

```bash
docker compose logs -f          # stream logs
docker compose down             # stop the container
docker compose up --build -d    # rebuild and restart after code changes
docker ps                       # check container status
```

---

## Bedrock Model IDs

Verify available Claude model IDs at **AWS Console → Amazon Bedrock → Model access**.

Default used: `us.anthropic.claude-sonnet-4-5-20250514-v1:0`  
Override via the `BEDROCK_PRIMARY_MODEL` environment variable.

---

## Project Structure

```
├── agents/
│   ├── intake_agent.py          input validation, metadata extraction, PII/profanity detection
│   ├── transcription_agent.py   Amazon Transcribe STT for audio files
│   ├── summarization_agent.py   LangChain + Pydantic: structured summary with PII/profanity suppression
│   ├── quality_score_agent.py   Claude function calling: QA rubric scoring
│   └── routing_agent.py         conditional routing functions and fallback/error nodes
├── graph/
│   ├── state.py                 AgentState TypedDict (single source of truth)
│   └── workflow.py              LangGraph StateGraph assembly with MemorySaver checkpointer
├── ui/
│   └── streamlit_app.py         4-tab Streamlit interface with call history sidebar
├── utils/
│   ├── llm_factory.py           Anthropic / Bedrock toggle (USE_BEDROCK env var)
│   ├── memory.py                SQLite call history persistence
│   ├── sensitive_data.py        regex-based PCI/PHI/PII detection + profanity word-list
│   ├── validation.py            Pydantic models: CallMetadata, CallSummary, QAScore
│   └── observability.py         log_agent decorator: timing, entry/exit, error logging
├── data/
│   └── sample_transcripts/      20 sample calls including PII/PHI/PCI, profanity, and mixed test files
├── .streamlit/
│   └── config.toml              theme, upload size, toolbar/top-bar settings
├── config/
│   └── mcp.yaml                 model control plane config
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Supported Input Formats

| Format | Description |
|--------|-------------|
| `.mp3`, `.wav`, `.m4a`, `.ogg` | Audio — transcribed via Amazon Transcribe |
| `.json` | Transcript with metadata: `agent_name`, `customer_name`, `duration_seconds`, `transcript` |
| `.txt` | Plain text transcript |

---

## QA Scoring Rubric

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Resolution | 35% | Was the issue resolved effectively? |
| Empathy | 25% | Did the agent show genuine understanding? |
| Professionalism | 20% | Language and conduct quality |
| Tone | 20% | Warmth and appropriateness throughout |
