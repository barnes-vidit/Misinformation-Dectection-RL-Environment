"""
FastAPI application for the Misinformation Detection Environment.

Follows the canonical chess_env / echo_env pattern exactly:
  - Single server, single port (7860 for HF Spaces)
  - task_id is selected per-episode via reset(task_id=...) kwarg
  - The default task_id is controlled by TASK_ID env var (default: "easy")
  - Clients can override per-episode by passing task_id in the reset body

Endpoints (provided by openenv-core create_app):
  POST /reset   — Reset with optional {task_id: "easy"|"medium"|"hard"}
  POST /step    — Submit FactCheckAction, returns observation with reward/done
  GET  /state   — Current FactCheckState (populated in WebSocket mode)
  GET  /health  — Health check
  WS   /ws      — WebSocket endpoint for stateful sessions (preserves state)

Usage:
  uvicorn lazarus.server.app:app --host 0.0.0.0 --port 7860
  TASK_ID=medium uvicorn lazarus.server.app:app --host 0.0.0.0 --port 7860
"""

import os

from openenv.core.env_server import create_app

from ..models import FactCheckAction, FactCheckObservation
from .environment import MisinformationEnvironment

# Default task at server startup — can be overridden per-episode in reset()
_default_task_id = os.getenv("TASK_ID", "easy")


def create_env() -> MisinformationEnvironment:
    """
    Factory called per session/request by the HTTP server.
    Creates a fresh env with the server-level default task_id.
    Clients can override task_id per-episode by passing it in the /reset body.
    """
    return MisinformationEnvironment(task_id=_default_task_id)


# Pass the factory function — matches chess_env / echo_env canonical pattern
app = create_app(
    create_env,
    FactCheckAction,
    FactCheckObservation,
    env_name="lazarus",
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for: uv run server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
