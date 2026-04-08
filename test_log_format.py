"""Smoke test for log format — no LLM calls, no network."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/lazarus")

# Stub openai so it doesn't try to connect
import unittest.mock as mock
sys.modules["openai"] = mock.MagicMock()

import lazarus.inference as inf
import inspect

src = inspect.getsource(inf)

checks = [
    ("START format",   'task={task_id} env=misinformation model={MODEL_NAME}' in src),
    ("STEP format",    'step={step} action={action} reward={reward:.2f}' in src),
    ("STEP done/err",  'done={str(done).lower()} error=null' in src),
    ("END format",     'success={str(success).lower()} steps={steps}' in src),
    ("END score",      'score={score:.2f} rewards={rewards_str}' in src),
    ("action=verdict", 'action=action_dict["verdict"]' in src),
    ("rewards_list",   'rewards_list' in src),
    ("normalized",     'normalized_score' in src),
]

all_ok = True
for name, ok in checks:
    status = "OK " if ok else "FAIL"
    print(f"  {status}  {name}")
    if not ok:
        all_ok = False

print()

# Dry-run the actual log functions
print("--- Sample output ---")
inf.log_start("easy")
inf.log_step(step=1, action="FALSE", reward=1.0, done=False)
inf.log_step(step=2, action="TRUE",  reward=0.0, done=False)
inf.log_step(step=3, action="MISLEADING", reward=0.5, done=True)
inf.log_end(success=True, steps=3, score=0.50, rewards=[1.0, 0.0, 0.5])

print()
print("ALL FORMAT CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
