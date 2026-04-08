"""Quick test: verifies done=True is sent on the final step when step_number is passed."""
import requests

base = 'http://localhost:7860'
max_steps = 3

r = requests.post(f'{base}/reset', json={}, timeout=10)
r.raise_for_status()
obs = r.json().get('observation', r.json())
print(f'Reset: step={obs["step_number"]}, max={obs["max_steps"]}, task={obs["task_id"]}')

action = {
    'verdict': 'FALSE',
    'confidence': 0.85,
    'reasoning': 'The article contradicts the claim directly.',
    'evidence_cited': ['contradicts', 'incorrect']
}

all_passed = True
for i in range(1, max_steps + 1):
    r = requests.post(
        f'{base}/step',
        json={'action': action, 'step_number': i},
        timeout=15,
    )
    r.raise_for_status()
    b = r.json()
    done = b['done']
    reward = b['reward']
    obs_step = b.get('observation', {}).get('step_number')
    print(f'  Step {i}/{max_steps}: reward={reward:.3f}  done={done}  next_step_in_obs={obs_step}')

    expected_done = (i >= max_steps)
    if done != expected_done:
        print(f'  FAIL: expected done={expected_done}, got done={done}')
        all_passed = False

    if done:
        print(f'  >>> done=True correctly signalled at step {i}')
        break

print()
print('DONE SIGNAL TEST:', 'PASSED' if all_passed else 'FAILED')
