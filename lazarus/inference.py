#!/usr/bin/env python3
"""
inference.py — Baseline fact-checking agent for the Misinformation Detection Environment.

Runs all three tasks (easy, medium, hard) in sequence using an OpenAI-compatible
LLM API, printing structured logs that judges can parse.

Architecture:
  Single server on port 7860 (HF Spaces standard).
  task_id is passed per-episode in the /reset request body — the server
  selects the correct difficulty dynamically. No multi-port needed.

Required environment variables:
  API_BASE_URL  — OpenAI-compatible API base URL (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model to use (e.g. gpt-4o-mini)
  HF_TOKEN      — Hugging Face token / OpenAI API key
  SERVER_URL    — Override env server URL (default: http://localhost:7860)

Run:
  API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py

Endpoints used:
  POST /reset  → body: {task_id: "easy"|"medium"|"hard"}  → {observation, reward, done}
  POST /step   → body: {action: <FactCheckAction fields>}  → {observation, reward, done}
  GET  /health → {status: "healthy"}
"""

from __future__ import annotations

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
SERVER_URL: str = os.environ.get("SERVER_URL", "http://localhost:7860")

TASKS: list[str] = ["easy", "medium", "hard"]
MAX_STEPS_BY_TASK: dict[str, int] = {"easy": 3, "medium": 5, "hard": 7}

SYSTEM_PROMPT: str = (
    "You are an expert fact-checker. Analyze the claim and article snippet carefully. "
    "Respond ONLY with valid JSON — no markdown, no code fences, no extra text.\n"
    'Format: {"verdict": str, "confidence": float, "reasoning": str, "evidence_cited": [str]}\n'
    "verdict must be EXACTLY one of: TRUE, FALSE, MISLEADING, UNVERIFIABLE\n"
    "confidence is a float between 0.0 (uncertain) and 1.0 (certain)\n"
    "reasoning is 1-3 sentences explaining your verdict\n"
    "evidence_cited is a list of short key phrases from the article that support your verdict"
)

RETRY_SYSTEM_PROMPT: str = (
    "You are a fact-checker. Respond ONLY with JSON, no extra text:\n"
    '{"verdict": "TRUE", "confidence": 0.8, "reasoning": "brief explanation", "evidence_cited": []}'
)

# ---------------------------------------------------------------------------
# OpenAI client initialisation
# ---------------------------------------------------------------------------

openai_client = OpenAI(
    api_key=HF_TOKEN or "placeholder",
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env=misinformation model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def http_reset(base_url: str, task_id: str) -> dict:
    """
    POST /reset with task_id so one server handles all 3 difficulties.
    Returns response JSON with 'observation', 'reward', 'done'.
    StepRequest has extra='allow', so task_id passes through cleanly.
    """
    resp = requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def http_step(base_url: str, action: dict) -> dict:
    """
    POST /step — action wrapped in {"action": {...}}.
    Returns response JSON with 'observation', 'reward', 'done'.
    """
    resp = requests.post(
        f"{base_url}/step",
        json={"action": action},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def http_health(base_url: str) -> bool:
    """GET /health — returns True if server is healthy."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("status") == "healthy"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------


def call_llm(
    claim: str,
    article_snippet: str,
    previous_feedback: str,
    retry: bool = False,
) -> dict:
    """
    Call the OpenAI-compatible API and parse the JSON response.
    Returns a dict with keys: verdict, confidence, reasoning, evidence_cited.
    Raises ValueError / json.JSONDecodeError on parse failure.
    """
    user_content = f"CLAIM: {claim}\n\nARTICLE SNIPPET:\n{article_snippet}"
    if previous_feedback:
        user_content += f"\n\nPREVIOUS FEEDBACK: {previous_feedback}"

    system = RETRY_SYSTEM_PROMPT if retry else SYSTEM_PROMPT

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    parsed = json.loads(raw_text)

    verdict = str(parsed.get("verdict", "UNVERIFIABLE")).strip().upper()
    if verdict not in {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}:
        verdict = "UNVERIFIABLE"

    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(parsed.get("reasoning", "No reasoning provided."))
    evidence_cited = [str(e) for e in parsed.get("evidence_cited", [])]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence_cited": evidence_cited,
    }


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------


def run_task(task_id: str, base_url: str) -> dict:
    """
    Run a complete episode for the given task_id against the single server.

    Passes task_id in the /reset request body — the server selects the
    correct task difficulty dynamically (no multi-port needed).

    Returns summary dict with final_score and avg_reward.
    """
    log_start(task_id)

    step_number = 0
    cumulative_reward = 0.0
    rewards_list: list[float] = []
    max_steps = MAX_STEPS_BY_TASK.get(task_id, 3)

    # Reset — passes task_id so server picks the right difficulty
    try:
        reset_resp = http_reset(base_url, task_id)
    except Exception as exc:
        print(f"[ERROR] Failed to reset for task '{task_id}': {exc}", file=sys.stderr)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"task_id": task_id, "final_score": 0.0, "avg_reward": 0.0}

    obs = reset_resp.get("observation", reset_resp)
    # Honour max_steps from the server observation (authoritative)
    max_steps = obs.get("max_steps", max_steps)

    current_obs = obs
    success = False

    for _ in range(max_steps):
        step_number += 1
        claim = current_obs.get("claim", "")
        article_snippet = current_obs.get("article_snippet", "")
        previous_feedback = current_obs.get("previous_feedback", "")

        # LLM call (one retry on parse error)
        try:
            action_dict = call_llm(claim, article_snippet, previous_feedback, retry=False)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            time.sleep(1)
            try:
                action_dict = call_llm(
                    claim, article_snippet, previous_feedback, retry=True
                )
            except Exception:
                action_dict = {
                    "verdict": "UNVERIFIABLE",
                    "confidence": 0.5,
                    "reasoning": "Could not parse LLM response.",
                    "evidence_cited": [],
                }

        # Submit action
        try:
            step_resp = http_step(base_url, action_dict)
        except Exception as exc:
            print(
                f"[ERROR] Step {step_number} failed for task '{task_id}': {exc}",
                file=sys.stderr,
            )
            break

        reward = float(step_resp.get("reward") or 0.0)
        cumulative_reward += reward
        rewards_list.append(reward)
        next_obs = step_resp.get("observation", {})
        done = bool(step_resp.get("done", False))

        log_step(
            step=step_number,
            action=action_dict["verdict"],
            reward=reward,
            done=done,
        )

        current_obs = next_obs

        if done:
            success = True
            break

    # Normalize score to [0, 1]: cumulative / max_possible (1.0 per step)
    max_possible = float(max_steps)
    normalized_score = cumulative_reward / max_possible if max_possible > 0 else 0.0
    normalized_score = max(0.0, min(1.0, normalized_score))

    log_end(success=success, steps=step_number, score=normalized_score, rewards=rewards_list)
    return {
        "task_id": task_id,
        "total_steps": step_number,
        "final_score": round(normalized_score, 4),
        "avg_reward": round(cumulative_reward / step_number if step_number > 0 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Server URL  : {SERVER_URL}", flush=True)
    print(f"Model       : {MODEL_NAME}", flush=True)
    print(f"API base URL: {API_BASE_URL}", flush=True)

    # Health check before starting
    if not http_health(SERVER_URL):
        print(f"[WARN] Server at {SERVER_URL} is not healthy. Proceeding anyway.", flush=True)

    print("-" * 60, flush=True)

    results = []
    for task_id in TASKS:
        result = run_task(task_id, SERVER_URL)
        results.append(result)
        print("-" * 60, flush=True)
        time.sleep(1)

    # Summary table
    print("\n=== BASELINE SUMMARY ===")
    print(f"{'Task':<10} {'Steps':>6} {'Score':>8} {'Avg Reward':>12}")
    print("-" * 40)
    for r in results:
        print(
            f"{r['task_id']:<10} {r.get('total_steps', 0):>6} "
            f"{r['final_score']:>8.4f} {r['avg_reward']:>12.4f}"
        )
    print("=" * 40)


if __name__ == "__main__":
    main()
