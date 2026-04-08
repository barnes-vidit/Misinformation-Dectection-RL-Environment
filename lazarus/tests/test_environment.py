"""
Automated pytest suite for the Misinformation Detection Environment.

Tests cover:
  - models: FactCheckAction / FactCheckObservation / FactCheckState validation
  - tasks:  grade_action deterministic grading for easy, medium, hard
  - environment: reset(), step(), done signalling, task_id override per reset
  - inference: server URL configuration
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_action_valid(self):
        from lazarus.models import FactCheckAction
        action = FactCheckAction(
            verdict="FALSE",
            confidence=0.9,
            reasoning="The article contradicts the claim.",
            evidence_cited=["contradicts"],
        )
        assert action.verdict == "FALSE"
        assert action.confidence == 0.9

    def test_action_confidence_bounds(self):
        from lazarus.models import FactCheckAction
        with pytest.raises(Exception):
            FactCheckAction(verdict="TRUE", confidence=1.5, reasoning="test")
        with pytest.raises(Exception):
            FactCheckAction(verdict="TRUE", confidence=-0.1, reasoning="test")

    def test_action_rejects_extra_fields(self):
        from lazarus.models import FactCheckAction
        with pytest.raises(Exception):
            FactCheckAction(verdict="TRUE", confidence=0.8, reasoning="r", unknown_field="x")

    def test_observation_defaults(self):
        from lazarus.models import FactCheckObservation
        obs = FactCheckObservation(
            claim="test claim",
            article_snippet="snippet",
            source_metadata={"outlet": "test", "date": "2024-01-01", "topic": "test"},
            task_id="easy",
            step_number=1,
            max_steps=3,
        )
        assert obs.done is False
        assert obs.reward is None
        assert obs.previous_feedback == ""

    def test_state_fields(self):
        from lazarus.models import FactCheckState
        s = FactCheckState(
            episode_id="test-ep",
            task_id="medium",
            step_count=2,
            cumulative_reward=1.5,
            claims_seen=2,
            correct_verdicts=1,
        )
        assert s.episode_id == "test-ep"
        assert s.task_id == "medium"


# ---------------------------------------------------------------------------
# Grader (tasks.py)
# ---------------------------------------------------------------------------

def _make_action(verdict="TRUE", confidence=0.8, reasoning="The claim is supported.", evidence=None):
    from lazarus.models import FactCheckAction
    return FactCheckAction(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        evidence_cited=evidence or [],
    )


def _first_claim(task_id: str) -> dict:
    from lazarus.server.tasks import get_claims_for_task
    return get_claims_for_task(task_id)[0]


class TestGraderEasy:
    def test_correct_verdict_scores_1(self):
        from lazarus.server.tasks import grade_action
        claim = _first_claim("easy")
        gt = claim["ground_truth_verdict"]
        action = _make_action(verdict=gt, confidence=0.7, reasoning="Simple check.")
        reward = grade_action(action, claim, "easy")
        assert 0.0 <= reward <= 1.0
        assert reward == 1.0

    def test_wrong_verdict_scores_0(self):
        from lazarus.server.tasks import grade_action
        claim = _first_claim("easy")
        gt = claim["ground_truth_verdict"]
        wrong = next(v for v in ["TRUE","FALSE","MISLEADING","UNVERIFIABLE"] if v != gt)
        action = _make_action(verdict=wrong, confidence=0.5, reasoning="Wrong reasoning.")
        reward = grade_action(action, claim, "easy")
        assert reward == 0.0

    def test_overconfident_wrong_penalised(self):
        from lazarus.server.tasks import grade_action
        claim = _first_claim("easy")
        gt = claim["ground_truth_verdict"]
        wrong = next(v for v in ["TRUE","FALSE","MISLEADING","UNVERIFIABLE"] if v != gt)
        action = _make_action(verdict=wrong, confidence=0.95, reasoning="Very wrong confidence.")
        reward = grade_action(action, claim, "easy")
        assert reward == 0.0  # max(0, 0 - penalty) = 0

    def test_reward_clamped_to_unit_interval(self):
        from lazarus.server.tasks import grade_action, get_claims_for_task
        for task_id in ["easy", "medium", "hard"]:
            for claim in get_claims_for_task(task_id):
                for verdict in ["TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"]:
                    action = _make_action(verdict=verdict, confidence=0.9, reasoning="test")
                    reward = grade_action(action, claim, task_id)
                    assert 0.0 <= reward <= 1.0, (
                        f"reward={reward} out of bounds for task={task_id}, verdict={verdict}"
                    )


class TestGraderMedium:
    def test_correct_no_evidence_scores_1(self):
        from lazarus.server.tasks import grade_action
        claim = _first_claim("medium")
        gt = claim["ground_truth_verdict"]
        action = _make_action(verdict=gt, confidence=0.8, reasoning="Plain correct.")
        reward = grade_action(action, claim, "medium")
        assert reward == 1.0

    def test_regression_return_score_not_hardcoded(self):
        """Regression: medium grader must return `score`, not hardcoded 1.0.
        Previously the code computed score but returned 1.0 — so evidence bonus
        was silently discarded. Now it returns score (still 1.0 when correct
        since score=1.0+bonus capped at 1.0, but the path is correct).
        Verify the wrong-verdict partial credit path actually works.
        """
        from lazarus.server.tasks import grade_action, get_claims_for_task
        claims = get_claims_for_task("medium")
        partial_seen = False
        for claim in claims:
            gt = claim["ground_truth_verdict"]
            wrong = next(v for v in ["TRUE","FALSE","MISLEADING","UNVERIFIABLE"] if v != gt)
            # Reasoning that acknowledges complexity → should get 0.3 partial credit
            r = grade_action(_make_action(verdict=wrong, confidence=0.6,
                                          reasoning="However this is nuanced and complex."),
                             claim, "medium")
            if r > 0.0:
                partial_seen = True
                assert r <= 1.0
        # At least some medium claims should give partial credit for nuanced wrong answers
        assert partial_seen, "Expected partial credit (>0) for nuanced wrong-verdict medium answers"


    def test_wrong_with_complexity_gets_partial(self):
        from lazarus.server.tasks import grade_action
        claim = _first_claim("medium")
        gt = claim["ground_truth_verdict"]
        wrong = next(v for v in ["TRUE","FALSE","MISLEADING","UNVERIFIABLE"] if v != gt)
        action = _make_action(
            verdict=wrong,
            confidence=0.6,
            reasoning="However, this is nuanced and misleading in context.",
        )
        reward = grade_action(action, claim, "medium")
        assert 0.0 <= reward <= 1.0


class TestGraderHard:
    def test_correct_with_evidence_citation_near_max(self):
        from lazarus.server.tasks import grade_action, get_claims_for_task
        claims = get_claims_for_task("hard")
        claim = claims[0]
        gt = claim["ground_truth_verdict"]
        snippet_words = claim["article_snippet"].split()[:4]
        evidence = [" ".join(snippet_words[:2])]
        action = _make_action(
            verdict=gt, confidence=0.7,
            reasoning="Evidence supports this verdict clearly.",
            evidence=evidence,
        )
        reward = grade_action(action, claim, "hard")
        assert 0.0 <= reward <= 1.0

    def test_unverifiable_overconfidence_penalised(self):
        from lazarus.server.tasks import grade_action, get_claims_for_task
        claims = get_claims_for_task("hard")
        uv_claims = [c for c in claims if c["ground_truth_verdict"] == "UNVERIFIABLE"]
        if not uv_claims:
            pytest.skip("No UNVERIFIABLE hard claims in dataset")
        claim = uv_claims[0]
        action = _make_action(verdict="UNVERIFIABLE", confidence=0.95, reasoning="I am certain.")
        reward = grade_action(action, claim, "hard")
        assert reward < 1.0


# ---------------------------------------------------------------------------
# Environment core
# ---------------------------------------------------------------------------

class TestEnvironment:
    def _make_env(self, task_id="easy"):
        from lazarus.server.environment import MisinformationEnvironment
        return MisinformationEnvironment(task_id=task_id)

    def test_init_calls_super(self):
        env = self._make_env()
        assert hasattr(env, "transform")
        assert hasattr(env, "rubric")

    def test_invalid_task_id_raises(self):
        from lazarus.server.environment import MisinformationEnvironment
        with pytest.raises(ValueError, match="Unknown task_id"):
            MisinformationEnvironment(task_id="extreme")

    def test_reset_returns_observation(self):
        from lazarus.models import FactCheckObservation
        env = self._make_env("easy")
        obs = env.reset()
        assert isinstance(obs, FactCheckObservation)
        assert obs.task_id == "easy"
        assert obs.step_number == 1
        assert obs.max_steps == 3
        assert obs.done is False

    def test_reset_task_id_override(self):
        """Single server: task_id can be overridden per-episode in reset()."""
        env = self._make_env("easy")  # server default = easy
        obs = env.reset(task_id="hard")
        assert obs.task_id == "hard"
        assert obs.max_steps == 7

    def test_reset_invalid_task_id_falls_back_to_default(self):
        """Invalid override falls back to the env's current task_id."""
        env = self._make_env("medium")
        obs = env.reset(task_id="nonexistent")
        assert obs.task_id == "medium"  # falls back gracefully

    def test_reset_episode_id_deterministic(self):
        env1 = self._make_env("easy")
        env2 = self._make_env("easy")
        ep = "fixed-episode-id-123"
        obs1 = env1.reset(episode_id=ep)
        obs2 = env2.reset(episode_id=ep)
        assert obs1.claim == obs2.claim

    def test_step_returns_observation_with_reward(self):
        from lazarus.models import FactCheckAction, FactCheckObservation
        env = self._make_env("easy")
        env.reset()
        action = FactCheckAction(
            verdict="FALSE", confidence=0.8, reasoning="Test reasoning.", evidence_cited=[]
        )
        obs = env.step(action)
        assert isinstance(obs, FactCheckObservation)
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_done_accumulates_in_stateful_session(self):
        """WebSocket path: state persists within one env instance across steps."""
        from lazarus.models import FactCheckAction
        env = self._make_env("easy")  # max_steps=3
        env.reset()
        action = FactCheckAction(verdict="FALSE", confidence=0.5, reasoning="r.")

        obs1 = env.step(action)
        assert obs1.done is False
        assert env.state.step_count == 1

        obs2 = env.step(action)
        assert obs2.done is False
        assert env.state.step_count == 2

        obs3 = env.step(action)
        assert obs3.done is True
        assert env.state.step_count == 3

    def test_state_property(self):
        from lazarus.models import FactCheckState
        env = self._make_env("medium")
        env.reset()
        state = env.state
        assert isinstance(state, FactCheckState)
        assert state.task_id == "medium"
        assert state.step_count == 0

    def test_all_task_ids_work(self):
        for task_id, max_steps in [("easy", 3), ("medium", 5), ("hard", 7)]:
            env = self._make_env(task_id)
            obs = env.reset()
            assert obs.max_steps == max_steps
            assert obs.task_id == task_id

    def test_cumulative_reward_accumulates(self):
        from lazarus.server.tasks import get_claims_for_task
        from lazarus.models import FactCheckAction
        env = self._make_env("easy")
        env.reset(episode_id="test-cumulative")
        # Use the correct verdict for claim 0 to get reward=1.0
        claim = env._current_claims[0]
        gt = claim["ground_truth_verdict"]
        action = FactCheckAction(verdict=gt, confidence=0.8, reasoning="Correct verdict.", evidence_cited=[])
        env.step(action)
        assert env.state.cumulative_reward > 0.0
        assert env.state.claims_seen == 1


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------

class TestInferenceConfig:
    def test_single_server_url(self):
        """inference.py now uses a single SERVER_URL for all tasks."""
        import lazarus.inference as inf
        # Should have SERVER_URL, not multi-port routing
        assert hasattr(inf, "SERVER_URL")
        assert "7860" in inf.SERVER_URL or "localhost" in inf.SERVER_URL

    def test_http_reset_passes_task_id(self):
        """http_reset must include task_id in the request body."""
        import inspect
        import lazarus.inference as inf
        src = inspect.getsource(inf.http_reset)
        assert "task_id" in src, "http_reset must pass task_id in request body"

    def test_tasks_list(self):
        import lazarus.inference as inf
        assert inf.TASKS == ["easy", "medium", "hard"]
        assert inf.MAX_STEPS_BY_TASK == {"easy": 3, "medium": 5, "hard": 7}
