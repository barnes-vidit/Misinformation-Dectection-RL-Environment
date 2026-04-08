"""
Misinformation Detection Environment — Python client.

Provides a typed client wrapping the openenv-core EnvClient with
FactCheckAction / FactCheckObservation / FactCheckState types.

Example (sync):
    from lazarus.client import FactCheckEnvClient, FactCheckAction
    with FactCheckEnvClient(base_url="http://localhost:7860") as client:
        result = client.reset()
        print(result.observation.claim)
        action = FactCheckAction(
            verdict="TRUE",
            confidence=0.85,
            reasoning="The article confirms the fact directly.",
            evidence_cited=["confirmed by multiple sources"],
        )
        result = client.step(action)
        print(result.reward)

Example (from Hugging Face Spaces):
    client = FactCheckEnvClient.from_hub("your-username/misinformation-env")
"""

from __future__ import annotations

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import FactCheckAction, FactCheckObservation, FactCheckState
except ImportError:
    from models import FactCheckAction, FactCheckObservation, FactCheckState


class FactCheckEnvClient(EnvClient[FactCheckAction, FactCheckObservation, FactCheckState]):
    """
    Typed client for the Misinformation Detection Environment.

    Inherits WebSocket-based communication from openenv-core EnvClient.
    Provides synchronous wrappers for convenience.

    Args:
        base_url: URL of the running environment server (e.g. "http://localhost:7860")
        task_id:  Task to run. One of "easy", "medium", "hard".
                  Passed as a query param on connect; the server uses TASK_ID env var
                  so this is informational only in HTTP mode.
    """

    def __init__(self, base_url: str = "http://localhost:7860", task_id: str = "easy") -> None:
        super().__init__(base_url=base_url)
        self.task_id = task_id

    # ------------------------------------------------------------------
    # from_hub classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_hub(cls, repo_id: str, task_id: str = "easy") -> "FactCheckEnvClient":
        """
        Connect to an environment deployed on Hugging Face Spaces.

        Args:
            repo_id: HF Spaces repo ID, e.g. "your-username/misinformation-env"
            task_id: Difficulty level — "easy", "medium", or "hard"

        Returns:
            A connected FactCheckEnvClient instance.

        Example:
            client = FactCheckEnvClient.from_hub("alice/misinformation-env", task_id="hard")
        """
        space_url = f"https://{repo_id.replace('/', '-')}.hf.space"
        return cls(base_url=space_url, task_id=task_id)

    # ------------------------------------------------------------------
    # Override _step_payload
    # ------------------------------------------------------------------

    def _step_payload(self, action: FactCheckAction) -> Dict:
        """Convert FactCheckAction to JSON-serialisable dict for the step message."""
        return {
            "verdict": action.verdict,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "evidence_cited": list(action.evidence_cited or []),
        }

    # ------------------------------------------------------------------
    # Override _parse_result
    # ------------------------------------------------------------------

    def _parse_result(self, payload: Dict) -> StepResult[FactCheckObservation]:
        """Parse server response into a typed StepResult[FactCheckObservation]."""
        obs_data = payload.get("observation", {})

        observation = FactCheckObservation(
            claim=obs_data.get("claim", ""),
            article_snippet=obs_data.get("article_snippet", ""),
            source_metadata=obs_data.get("source_metadata", {"outlet": "", "date": "", "topic": ""}),
            task_id=obs_data.get("task_id", self.task_id),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 3),
            previous_feedback=obs_data.get("previous_feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    # ------------------------------------------------------------------
    # Override _parse_state
    # ------------------------------------------------------------------

    def _parse_state(self, payload: Dict) -> FactCheckState:
        """Parse server state payload into a FactCheckState."""
        return FactCheckState(
            episode_id=payload.get("episode_id", "unknown"),
            task_id=payload.get("task_id", self.task_id),
            step_count=payload.get("step_count", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            claims_seen=payload.get("claims_seen", 0),
            correct_verdicts=payload.get("correct_verdicts", 0),
        )
