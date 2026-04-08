"""
Data models for the Misinformation Detection Environment (Lazarus).

from openenv.core.env_server.types import Action, Observation, State
(re-exported from openenv.core.env_server for convenience)
"""

from __future__ import annotations

from typing import List

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class FactCheckAction(Action):
    """Action submitted by agent: a verdict with reasoning and evidence."""

    verdict: str = Field(
        ...,
        description="One of: TRUE, FALSE, MISLEADING, UNVERIFIABLE",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its verdict (0.0 = uncertain, 1.0 = certain)",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of verdict (1-3 sentences)",
    )
    evidence_cited: List[str] = Field(
        default_factory=list,
        description="Key phrases from article used as evidence",
    )


class FactCheckObservation(Observation):
    """Observation received by the agent: a claim + article snippet to analyse."""

    claim: str = Field(
        ...,
        description="The headline/claim to fact-check",
    )
    article_snippet: str = Field(
        ...,
        description="Supporting article text (150-300 words)",
    )
    source_metadata: dict = Field(
        ...,
        description="{'outlet': str, 'date': str, 'topic': str}",
    )
    task_id: str = Field(
        ...,
        description="easy | medium | hard",
    )
    step_number: int = Field(
        ...,
        description="Current step in episode",
    )
    max_steps: int = Field(
        ...,
        description="Max steps for this task (3/5/7)",
    )
    previous_feedback: str = Field(
        default="",
        description="Feedback from last step if any",
    )


class FactCheckState(State):
    """Persistent state tracked across an episode."""

    episode_id: str = Field(
        default="",
        description="Unique episode identifier",
    )
    task_id: str = Field(
        default="easy",
        description="Task difficulty: easy | medium | hard",
    )
    step_count: int = Field(
        default=0,
        description="Steps completed so far",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of rewards earned",
    )
    claims_seen: int = Field(
        default=0,
        description="Number of claims presented",
    )
    correct_verdicts: int = Field(
        default=0,
        description="Number of verdicts that exactly matched ground truth",
    )
