"""
End-to-end validation: tests all 3 task servers (easy/medium/hard) simultaneously.
Verifies health, reset, step mechanics, reward bounds, and done signalling per task.
"""
import requests

TASK_SERVERS = {
    "easy":   ("http://localhost:7860", 3),
    "medium": ("http://localhost:7861", 5),
    "hard":   ("http://localhost:7862", 7),
}

action = {
    "verdict": "FALSE",
    "confidence": 0.85,
    "reasoning": "The article directly contradicts this claim with specific evidence.",
    "evidence_cited": ["contradicts", "specific evidence"],
}

all_passed = True

for task_id, (url, max_steps) in TASK_SERVERS.items():
    print(f"\n{'='*50}")
    print(f"Task: {task_id.upper()} ({url})")

    # 1. Health
    try:
        r = requests.get(f"{url}/health", timeout=5)
        assert r.status_code == 200 and r.json().get("status") == "healthy"
        print(f"  ✓ /health OK")
    except Exception as e:
        print(f"  ✗ /health FAILED: {e}")
        all_passed = False
        continue

    # 2. Reset
    r = requests.post(f"{url}/reset", json={}, timeout=10)
    assert r.status_code == 200, f"reset failed: {r.status_code}"
    obs = r.json().get("observation", r.json())
    assert obs["task_id"] == task_id, f"task_id mismatch: got {obs['task_id']}"
    assert obs["max_steps"] == max_steps, f"max_steps: got {obs['max_steps']} expected {max_steps}"
    assert obs["step_number"] == 1
    print(f"  ✓ /reset OK — claim: {obs['claim'][:55]}...")

    # 3. Full episode
    rewards = []
    done_at = None
    for step_n in range(1, max_steps + 1):
        r = requests.post(
            f"{url}/step",
            json={"action": action, "step_number": step_n},
            timeout=15,
        )
        assert r.status_code == 200, f"step {step_n} failed: {r.status_code}"
        b = r.json()
        reward = b.get("reward", 0.0)
        done = b.get("done", False)
        assert 0.0 <= reward <= 1.0, f"reward {reward} out of bounds at step {step_n}"
        rewards.append(reward)
        if done:
            done_at = step_n

    assert done_at == max_steps, f"done_at={done_at} expected {max_steps}"
    avg = sum(rewards) / len(rewards)
    print(f"  ✓ Full episode OK — {max_steps} steps, avg_reward={avg:.3f}, done at step {done_at}")

print(f"\n{'='*50}")
print("ALL E2E CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
