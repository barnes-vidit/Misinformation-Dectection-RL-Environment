import requests, json, time

base = 'http://localhost:7860'

# 1. Health
r = requests.get(f'{base}/health', timeout=5)
assert r.status_code == 200, f'Health failed: {r.status_code}'
print('OK  /health:', r.json())

# 2. Reset
r = requests.post(f'{base}/reset', json={}, timeout=10)
assert r.status_code == 200, f'Reset failed: {r.status_code}'
body = r.json()
obs = body.get('observation', body)
print('OK  /reset:')
print('      claim      :', obs.get('claim','')[:70])
print('      task_id    :', obs.get('task_id'))
print('      step_number:', obs.get('step_number'))
print('      max_steps  :', obs.get('max_steps'))
print('      done       :', body.get('done'))
assert obs.get('task_id') == 'easy', f"task_id mismatch: {obs.get('task_id')}"
assert obs.get('max_steps') == 3, f"max_steps should be 3: {obs.get('max_steps')}"
assert obs.get('step_number') == 1, f"step_number should be 1: {obs.get('step_number')}"

# 3. Step with wrapped action (AGENTS.md: POST /step with {action: {...}})
action = {
    'action': {
        'verdict': 'FALSE',
        'confidence': 0.85,
        'reasoning': 'The article states Paris France not Berlin so the claim is wrong.',
        'evidence_cited': ['Paris, France', 'Champ de Mars']
    }
}
t0 = time.time()
r = requests.post(f'{base}/step', json=action, timeout=15)
elapsed = time.time() - t0
assert r.status_code == 200, f'Step failed ({elapsed:.1f}s): {r.status_code} {r.text[:200]}'
body = r.json()
obs2 = body.get('observation', {})
reward = body.get('reward', -1)
done = body.get('done')
print(f'OK  /step ({elapsed:.2f}s):')
print(f'      reward     : {reward}')
print(f'      done       : {done}')
print(f'      step_number: {obs2.get("step_number")}')
assert 0.0 <= reward <= 1.0, f'Reward {reward} not in [0,1]'

# 4. State
r = requests.get(f'{base}/state', timeout=5)
assert r.status_code == 200, f'State failed: {r.status_code}'
state = r.json()
print('OK  /state:', {k: state.get(k) for k in ['step_count', 'cumulative_reward', 'task_id', 'episode_id']})

# 5. Full episode: reset then 3 steps until done=True
r = requests.post(f'{base}/reset', json={}, timeout=10)
print('OK  Full episode test:')
for i in range(3):
    r = requests.post(f'{base}/step', json=action, timeout=15)
    b = r.json()
    print(f'  step {i+1}: reward={b.get("reward"):.3f}  done={b.get("done")}')

# 6. Medium task (TASK_ID override not available but test step logic)
print()
print('ALL VALIDATION CHECKS PASSED')
