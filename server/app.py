"""
server/app.py — IncidentMind FastAPI HTTP Server
================================================
OWNER: Ritu
PORT: 7860 (mandatory — Amrita's Dockerfile exposes this port)
DEPENDS ON: models.py, server/environment.py

Endpoints:
  POST /reset   → Accept ResetRequest, return Observation
  POST /step    → Accept Action, return StepResult
  GET  /state   → Return State
  GET  /health  → Return {"status": "ok"}  (used by CI + HF Spaces)

Key design decisions:
  - Single IncidentEnvironment instance shared across all requests (lifespan)
  - CORS enabled for all origins (required for HF Space web interface)
  - All handlers are async def
  - HTTP 422 with clear message for invalid action parameters
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request


from fastapi.middleware.cors import CORSMiddleware

from IncidentMind.models import Action, Observation, ResetRequest, State, StepResult
from IncidentMind.server.environment import IncidentEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: create ONE shared environment at startup, dispose at shutdown
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    The environment is created ONCE when the server starts.
    All requests share the same instance — this is intentional.
    """
    app.state.env = IncidentEnvironment()
    print("[IncidentMind] Environment initialised and ready.")
    yield
    # Cleanup (nothing needed here, but the pattern requires yield)
    print("[IncidentMind] Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App definition
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IncidentMind",
    description=(
        "AI-powered SRE incident triage environment for the "
        "Meta PyTorch × Hugging Face OpenEnv Hackathon 2026. "
        "Agents learn to identify root causes, trigger runbooks, "
        "and resolve incidents across three difficulty tiers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS: allow requests from any origin (required for HF Space frontend) ────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helper — avoids repeating app.state.env everywhere
# ─────────────────────────────────────────────────────────────────────────────
def get_env(request) -> IncidentEnvironment:
    return request.app.state.env


# ─────────────────────────────────────────────────────────────────────────────
# GET /health  — used by CI pipeline and HF Spaces health check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """
    Static health check. Returns 200 + {"status": "ok"} when the server is up.
    Amrita's CI pipeline curls this after docker run to confirm startup.
    """
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# GET /      — root endpoint to prevent 404s on HF Spaces
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["ops"])
async def root() -> dict[str, str]:
    """Root endpoint to show the API is running."""
    return {"message": "Welcome to IncidentMind API! The environment is running.", "endpoints": ["/health", "/reset", "/step", "/state"]}



# ─────────────────────────────────────────────────────────────────────────────
# POST /reset  — start a new episode
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/reset", response_model=Observation, tags=["environment"])
async def reset(body: ResetRequest, request: Request) -> Observation:

    """
    Start a new episode for the given task.

    Request body:
        {"task_id": "task1", "seed": 42}
        {"task_id": "task2"}          ← seed is optional

    Returns:
        Full Observation Pydantic model as JSON.

    Raises:
        422 if task_id is not one of: task1, task2, task3
        (Pydantic validates this automatically via the Literal type in ResetRequest)
    """
    env: IncidentEnvironment = get_env(request)
    observation = env.reset(task_id=body.task_id, seed=body.seed)
    return observation


# ─────────────────────────────────────────────────────────────────────────────
# POST /step  — apply one agent action
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/step", response_model=StepResult, tags=["environment"])
async def step(action: Action, request: Request) -> StepResult:
    """
    Apply one action to the current episode.

    Request body examples:
        {"action_type": "INVESTIGATE", "parameters": {"alert_id": "alert-001"}}
        {"action_type": "TRIGGER_RUNBOOK", "parameters": {"runbook_id": "RUNBOOK_DB_POOL_RESET"}}
        {"action_type": "RESOLVE", "parameters": {}}

    Returns:
        StepResult with new Observation, reward, done flag, and info dict.
        When done=True, info["grade_result"] contains the final GradeResult dict.

    Raises:
        422 if action_type is not one of the 7 valid strings (Pydantic validates this).
        422 if required parameters are missing (handled in environment._handle_* methods,
            reflected in info["error"] with reward=-0.05).
    """
    env: IncidentEnvironment = get_env(request)

    # Pydantic already validated action_type via the Literal type in Action model.
    # If we reach here, action_type is guaranteed to be one of the 7 valid strings.
    result = env.step(action)

    # If environment returned an error in info (e.g., episode already done), still
    # return 200 with the StepResult — the client reads info["error"].
    # Only raise HTTP errors for true server-side failures.
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /state  — lightweight read of episode metadata
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/state", response_model=State, tags=["environment"])
async def state(request: Request) -> State:
    """
    Returns lightweight episode state (episode_id, step_count, elapsed_seconds, task_id).
    Does NOT return the full Observation. Does NOT change environment state.

    Raises:
        503 if no episode has been started yet (episode_id is empty).
    """
    env: IncidentEnvironment = get_env(request)

    if not env._episode_id:
        raise HTTPException(
            status_code=503,
            detail="No episode is active. Call POST /reset first.",
        )

    return env.state()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for local development (not used by Amrita's Dockerfile)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # For local dev only. Amrita's Dockerfile uses:
    #   CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)
