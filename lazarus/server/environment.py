"""
MisinformationEnvironment — core OpenEnv environment for fact-checking tasks.

Follows the canonical chess_env pattern exactly:
  - One server, one port (7860 for HF Spaces)
  - task_id can be set at construction time (from TASK_ID env var via create_env())
    OR overridden per-episode at reset() time by passing task_id as a kwarg
  - State persists within a session (WebSocket) or within a single stateless call
  - The /ws endpoint gives full stateful sessions; /reset+/step are stateless helpers

WebSocket vs HTTP:
  - WebSocket (/ws): create_app keeps one env instance per connection alive.
    State accumulates correctly across steps — this is the canonical usage.
  - HTTP (/reset, /step): a fresh env is created per call (openenv-core behaviour).
    The env is stateless across HTTP calls; inference.py uses the WebSocket client
    (FactCheckEnvClient) for RL training, or the HTTP endpoints for simple testing.
"""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from ..models import FactCheckAction, FactCheckObservation, FactCheckState
from .tasks import get_claims_for_task, grade_action

# Maximum steps per task difficulty
_MAX_STEPS: dict[str, int] = {"easy": 3, "medium": 5, "hard": 7}
_VALID_TASKS = set(_MAX_STEPS)


class MisinformationEnvironment(Environment):
    """
    An environment where an AI agent fact-checks fictional news claims.

    Each episode consists of multiple steps. At each step the agent
    receives a claim + article snippet and must return a verdict with
    reasoning. Rewards are computed by the deterministic grader in tasks.py.

    task_id controls difficulty and can be overridden per-episode in reset():
        env.reset(task_id="hard")

    step() returns a FactCheckObservation — reward and done are fields on
    the observation object (openenv-core convention).
    """

    def __init__(self, task_id: str = "easy") -> None:
        super().__init__()  # sets self.transform = None, self.rubric = None
        if task_id not in _VALID_TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Must be one of {sorted(_VALID_TASKS)}")
        self.task_id = task_id
        self._max_steps = _MAX_STEPS[task_id]
        self._state: FactCheckState | None = None
        self._current_claims: list[dict] = []
        self._claim_index: int = 0
        self._last_feedback: str = ""

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> FactCheckObservation:
        """
        Reset the environment and return the first observation.

        Args:
            seed: Random seed (unused; episode_id provides determinism).
            episode_id: Fixed episode ID for reproducible claim ordering.
            task_id: Override task difficulty for this episode.
                     One of "easy", "medium", "hard". Defaults to self.task_id.
        """
        try:
            # Allow per-episode task_id override — key for single-server multi-task
            active_task = task_id if (task_id and task_id in _VALID_TASKS) else self.task_id

            ep_id = episode_id or str(uuid4())
            claims = get_claims_for_task(active_task)

            # Deterministic shuffle per episode via episode_id seed
            rng = random.Random(ep_id)
            rng.shuffle(claims)

            self.task_id = active_task  # update for state/step consistency
            self._max_steps = _MAX_STEPS[active_task]
            self._current_claims = claims
            self._claim_index = 0
            self._last_feedback = ""

            self._state = FactCheckState(
                episode_id=ep_id,
                task_id=active_task,
                step_count=0,
                cumulative_reward=0.0,
                claims_seen=0,
                correct_verdicts=0,
            )

            return self._build_observation(
                step_number=1,
                feedback="",
                reward=0.0,
                done=False,
            )
        except Exception as exc:  # noqa: BLE001
            return self._fallback_observation(str(exc))

    def step(
        self,
        action: FactCheckAction,
        timeout_s: float | None = None,
        episode_id: str | None = None,
        step_index: int | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> FactCheckObservation:
        """
        Execute one step: grade the action, update state, return next observation.

        HTTP stateless mode: openenv-core creates a fresh env per /step call.
        The client passes back episode_id + step_index + task_id so we can
        reconstruct the exact claim list (deterministic shuffle from episode_id)
        and jump to the correct claim position — no server-side session needed.

        WebSocket mode: state persists normally across steps.
        """
        try:
            # ---------------------------------------------------------------
            # Stateless HTTP reconstruction
            # If called on a fresh env (self._state is None) AND the client
            # provided episode context, reconstruct deterministically.
            # ---------------------------------------------------------------
            if self._state is None or not self._current_claims:
                if episode_id is not None and step_index is not None:
                    # Use client-supplied task_id if valid
                    active_task = (
                        task_id if (task_id and task_id in _VALID_TASKS) else self.task_id
                    )
                    claims = get_claims_for_task(active_task)
                    rng = random.Random(episode_id)
                    rng.shuffle(claims)

                    self.task_id = active_task
                    self._max_steps = _MAX_STEPS[active_task]
                    self._current_claims = claims
                    self._claim_index = step_index  # jump to the right position
                    self._last_feedback = ""
                    self._state = FactCheckState(
                        episode_id=episode_id,
                        task_id=active_task,
                        step_count=step_index,
                        cumulative_reward=0.0,
                        claims_seen=step_index,
                        correct_verdicts=0,
                    )
                else:
                    # Fallback: no context provided, reset with defaults
                    self.reset()

            current_claim = self._current_claims[self._claim_index]
            reward = grade_action(action, current_claim, self.task_id)
            reward = float(max(0.0, min(1.0, reward)))

            # Update state — this persists in WebSocket sessions
            self._state.step_count += 1
            self._state.cumulative_reward += reward
            self._state.claims_seen += 1

            verdict_upper = (action.verdict or "").strip().upper()
            if verdict_upper == current_claim["ground_truth_verdict"]:
                self._state.correct_verdicts += 1

            done = self._state.step_count >= self._max_steps

            # Build feedback
            correct_verdict = current_claim["ground_truth_verdict"]
            if verdict_upper == correct_verdict:
                feedback = (
                    f"Correct! The verdict was {correct_verdict}. "
                    f"{current_claim['explanation']}"
                )
            else:
                feedback = (
                    f"Incorrect. Your verdict: {verdict_upper}. "
                    f"Correct verdict: {correct_verdict}. "
                    f"{current_claim['explanation']}"
                )

            if done:
                feedback += (
                    f" | Episode complete. "
                    f"Score: {self._state.cumulative_reward:.3f}/{self._max_steps:.0f}."
                )

            self._last_feedback = feedback

            # Advance claim index (cycles through dataset)
            self._claim_index = (self._claim_index + 1) % len(self._current_claims)

            next_step = self._state.step_count + 1
            return self._build_observation(
                step_number=next_step if not done else self._state.step_count,
                feedback=feedback,
                reward=reward,
                done=done,
            )

        except Exception as exc:  # noqa: BLE001
            return self._fallback_observation(f"step() error: {exc}")


    @property
    def state(self) -> FactCheckState:
        """Current episode state. Fully populated in WebSocket sessions."""
        if self._state is None:
            return FactCheckState(
                episode_id="uninitialized",
                task_id=self.task_id,
                step_count=0,
                cumulative_reward=0.0,
                claims_seen=0,
                correct_verdicts=0,
            )
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        step_number: int,
        feedback: str,
        reward: float,
        done: bool,
    ) -> FactCheckObservation:
        """Build a FactCheckObservation from the current claim."""
        claim_dict = self._current_claims[self._claim_index]
        return FactCheckObservation(
            claim=claim_dict["claim"],
            article_snippet=claim_dict["article_snippet"],
            source_metadata=claim_dict["source_metadata"],
            task_id=self.task_id,
            step_number=step_number,
            max_steps=self._max_steps,
            previous_feedback=feedback,
            done=done,
            reward=reward,
        )

    def _fallback_observation(self, error_msg: str) -> FactCheckObservation:
        """Safe fallback observation used when an exception occurs."""
        if not self._current_claims:
            try:
                self._current_claims = get_claims_for_task(self.task_id)
            except Exception:
                self._current_claims = [{
                    "claim": "[error]",
                    "article_snippet": error_msg,
                    "source_metadata": {"outlet": "system", "date": "N/A", "topic": "error"},
                    "ground_truth_verdict": "UNVERIFIABLE",
                    "explanation": "Internal error.",
                }]
            self._claim_index = 0

        return FactCheckObservation(
            claim="[Environment error — please reset]",
            article_snippet=f"An internal error occurred: {error_msg}",
            source_metadata={"outlet": "system", "date": "N/A", "topic": "error"},
            task_id=self.task_id,
            step_number=0,
            max_steps=self._max_steps,
            previous_feedback="",
            done=True,
            reward=0.0,
        )
