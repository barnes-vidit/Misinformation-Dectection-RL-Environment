# Misinformation Detection Environment

> **OpenEnv Hackathon Submission** — Meta × Hugging Face OpenEnv AI Hackathon (India, April 2026)

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates a real-world **fact-checking desk**. An AI agent receives fictional news headlines with supporting article snippets and must classify each claim as:

- `TRUE` — the claim is accurate based on the article
- `FALSE` — the claim is factually incorrect
- `MISLEADING` — technically true but missing critical context or cherry-picked
- `UNVERIFIABLE` — insufficient information in the article to make a determination

This is a genuinely novel OpenEnv domain with immediate real-world value: teaching AI agents to reason carefully about evidence, resist overconfidence, and distinguish between outright falsehoods and subtly misleading statistics or misattributed quotes.

---

## Why Misinformation Detection?

The proliferation of misinformation poses one of the most significant challenges to democratic societies. Automated tools that can identify misleading content at scale — while also calibrating their confidence — are urgently needed. This environment trains agents to:

1. **Read carefully** rather than pattern-match on surface features
2. **Distinguish nuance** between FALSE, MISLEADING, and UNVERIFIABLE
3. **Cite evidence** from the source text
4. **Calibrate confidence** appropriately (overconfidence is penalised)

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `str` | One of: `TRUE`, `FALSE`, `MISLEADING`, `UNVERIFIABLE` |
| `confidence` | `float` | Agent's certainty, range `0.0` (uncertain) → `1.0` (certain) |
| `reasoning` | `str` | 1–3 sentence justification of the verdict |
| `evidence_cited` | `list[str]` | Key phrases extracted from the article snippet |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The news headline or claim to fact-check |
| `article_snippet` | `str` | 150–300 words of contextual article text |
| `source_metadata` | `dict` | `{outlet: str, date: str, topic: str}` |
| `task_id` | `str` | Difficulty level: `easy` \| `medium` \| `hard` |
| `step_number` | `int` | Current step index (1-indexed) |
| `max_steps` | `int` | Maximum steps for this difficulty level |
| `previous_feedback` | `str` | Feedback from the previous step; empty on first step |

---

## Tasks

| Task | `task_id` | Steps | Description |
|------|-----------|-------|-------------|
| **Easy** | `easy` | 3 | Clearly true or false claims with obvious clues in the snippet. Straightforward factual errors and direct contradictions. |
| **Medium** | `medium` | 5 | Nuanced claims that require careful reading. `MISLEADING` verdicts are common — think cherry-picked statistics, misattributed quotes, and partial truths. |
| **Hard** | `hard` | 7 | Complex multi-evidence claims. Some are `UNVERIFIABLE`. Overconfidence on uncertain claims is penalised. Evidence citation is rewarded. |

---

## Reward Function

All rewards are deterministic floats in `[0.0, 1.0]`. Same inputs always produce the same reward.

### Easy Task
| Condition | Reward |
|-----------|--------|
| Correct verdict | `+1.0` |
| Correct verdict + key evidence phrase in reasoning | `min(1.0, +1.0 + 0.3) = 1.0` (capped) |
| Wrong verdict, confidence > 0.9 | `-0.2` (overconfidence penalty, floored at 0.0) |
| Wrong verdict, confidence ≤ 0.9 | `0.0` |

### Medium Task
| Condition | Reward |
|-----------|--------|
| Exact verdict match | `1.0` |
| Wrong verdict + reasoning acknowledges complexity + evidence cited | `0.6` |
| Wrong verdict + reasoning acknowledges complexity only | `0.3` |
| Wrong verdict + evidence cited only | `0.3` |
| Wrong verdict, no nuance | `0.0` |

*Complexity acknowledgement: reasoning contains words like "however", "misleading", "context", "partial", "cherry", "nuanced".*

### Hard Task
| Condition | Reward |
|-----------|--------|
| Correct verdict | `1.0` |
| Correct verdict + evidence cited | `min(1.0, 1.0 + 0.3) = 1.0` (capped) |
| Correct UNVERIFIABLE + confidence > 0.8 | `1.0 - 0.3 = 0.7` (overconfidence penalty) |
| Wrong verdict + evidence cited + reasoning ≥ 30 chars | `0.3` (minus any overconfidence) |
| Wrong verdict, no engagement | `0.0` |

---

## Setup

### Prerequisites
- Python 3.10+
- `pip` or `uv`

### Install from source

```bash
git clone https://github.com/your-username/misinformation-env
cd misinformation-env/lazarus
pip install -e .
```

### Run the server

```bash
# Default task: easy (port 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860

# With uv
uv run server

# Change difficulty
TASK_ID=medium uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
# Build
docker build -t misinformation-env .

# Run (easy task)
docker run -p 7860:7860 misinformation-env

# Run (hard task)
docker run -p 7860:7860 -e TASK_ID=hard misinformation-env
```

### Verify installation

```bash
curl http://localhost:7860/health
# → {"status": "ok"}

curl -X POST http://localhost:7860/reset
# → {"observation": {"claim": "...", "article_snippet": "...", ...}, ...}

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"verdict":"TRUE","confidence":0.8,"reasoning":"The article confirms it directly.","evidence_cited":["confirmed in the text"]}'
```

---

## Running Inference

```bash
# With OpenAI
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=hf_your_token \
python inference.py

# With a local model (e.g. LM Studio)
API_BASE_URL=http://localhost:1234/v1 \
MODEL_NAME=local-model \
HF_TOKEN=dummy \
python inference.py
```

The script runs all three tasks in sequence and prints structured logs:

```
[START] {"task_id": "easy", "model": "gpt-4o-mini", "timestamp": "2026-04-07T00:00:00+00:00"}
[STEP]  {"task_id": "easy", "step": 1, "claim": "...", "verdict": "FALSE", "reward": 1.0, "cumulative_reward": 1.0}
[STEP]  {"task_id": "easy", "step": 2, ...}
[STEP]  {"task_id": "easy", "step": 3, ...}
[END]   {"task_id": "easy", "total_steps": 3, "final_score": 2.7, "avg_reward": 0.9}
```

---

## Deploy to Hugging Face Spaces

```bash
# Login
huggingface-cli login

# Push (requires openenv CLI)
openenv push --repo-id your-username/misinformation-env
```

---

## Python Client

```python
from lazarus.client import FactCheckEnvClient
from lazarus.models import FactCheckAction

# Local server
with FactCheckEnvClient(base_url="http://localhost:7860") as client:
    result = client.reset()
    obs = result.observation
    print(obs.claim)

    action = FactCheckAction(
        verdict="MISLEADING",
        confidence=0.75,
        reasoning="The article shows the figure is a one-time inflow, not sustained growth.",
        evidence_cited=["one-time inflow", "statistical artefact"],
    )
    result = client.step(action)
    print(f"Reward: {result.reward}")

# From Hugging Face Spaces
client = FactCheckEnvClient.from_hub("your-username/misinformation-env", task_id="hard")
```

---

## Baseline Results (Llama-3.1-8B-Instruct)

| Task   |Steps | Score  | Avg Reward |
|--------|------|--------|------------|
| easy   | 3    | 0.3333 | 0.3333     |
| medium | 5    | 0.4000 | 0.4000     |
| hard   | 7    | 0.5714 | 0.5714     |

---

## Example Interaction

**Claim:**
> "Greenland's GDP grew by 40% last year, proving its economic miracle."

**Article snippet (excerpt):**
> *"Economists caution that the figure is highly misleading: the contract revenue is a one-time inflow and most profits will repatriate to foreign shareholders. Unemployment in Greenland actually rose slightly... Analysts described the growth figure as a 'statistical artefact.'"*

**Agent response:**
```json
{
  "verdict": "MISLEADING",
  "confidence": 0.82,
  "reasoning": "The 40% GDP gain reflects a single one-time foreign oil contract, not sustained economic growth. Unemployment rose and analysts called the figure a statistical artefact.",
  "evidence_cited": ["one-time inflow", "statistical artefact", "unemployment actually rose"]
}
```

**Reward: `1.0`** (correct verdict; key evidence phrases present in reasoning)

---

## File Structure

```
lazarus/
├── models.py            ← FactCheckAction, FactCheckObservation, FactCheckState
├── client.py            ← FactCheckEnvClient (typed WebSocket client)
├── inference.py         ← Baseline OpenAI agent (judges run this)
├── openenv.yaml         ← Environment metadata
├── Dockerfile           ← HF Spaces compatible (port 7860)
├── requirements.txt
├── pyproject.toml
├── README.md
└── server/
    ├── __init__.py
    ├── tasks.py         ← 18 fictional claims + deterministic grader
    ├── environment.py   ← MisinformationEnvironment (reset/step/state)
    └── app.py           ← FastAPI server via create_app()
```

---

## License

BSD 3-Clause — see `LICENSE` file for details.
