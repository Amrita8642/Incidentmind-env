"""
client.py — IncidentMind OpenEnv HTTP Client
============================================
OWNER: Ritu
USED BY: inference.py, external agents, hackathon judges

This wraps the raw HTTP calls to the FastAPI server so that agent code
never writes requests.post() directly — it uses this clean interface instead.

Usage (async):
    async with IncidentEnvClient("http://localhost:7860") as client:
        obs = await client.reset("task1", seed=42)
        result = await client.step({"action_type": "INVESTIGATE",
                                    "parameters": {"alert_id": "alert-001"}})

Usage (sync wrapper — for scripts that are not async):
    client = IncidentEnvClient("http://localhost:7860")
    obs = client.sync_reset("task1", seed=42)
    result = client.sync_step({"action_type": "RESOLVE", "parameters": {}})

NOTE on openenv-core: The document says to inherit from openenv-core's EnvClient.
If openenv-core does not yet have an importable base class, implement the methods
directly. The interface contract is what matters for judging — not the inheritance.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from models import Action, Observation, State, StepResult


# ─────────────────────────────────────────────────────────────────────────────
# Try to inherit from openenv-core if available; fall back gracefully
# ─────────────────────────────────────────────────────────────────────────────
try:
    from openenv.client import EnvClient as _OpenEnvBase  # type: ignore[import]
    _BASE = _OpenEnvBase
    _HAS_OPENENV = True
except ImportError:
    # openenv-core not installed or base class path differs
    # Remove this fallback once openenv-core is confirmed installed
    _BASE = object  # type: ignore[assignment,misc]
    _HAS_OPENENV = False


class IncidentEnvClient(_BASE):  # type: ignore[misc]
    """
    HTTP client for the IncidentMind FastAPI server.

    All async methods use httpx.AsyncClient internally.
    Sync wrappers (sync_reset, sync_step, sync_state) run the async
    coroutines with asyncio.run() so they work in plain scripts.
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        """
        Args:
            base_url: Full URL of the running IncidentMind server.
                      Default is localhost:7860 for local dev.
                      In inference.py, read this from API_BASE_URL env var.
        """
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ── Async context manager ─────────────────────────────────────────────────
    async def __aenter__(self) -> "IncidentEnvClient":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the async client, creating one if not inside a context manager."""
        if self._client is None:
            # If used outside async with, create a persistent client
            # Caller is responsible for closing it (or use sync_* methods)
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self._client

    # ── Core async methods ────────────────────────────────────────────────────
    async def reset(
        self, task_id: str, seed: Optional[int] = None
    ) -> Observation:
        """
        POST /reset — start a new episode.

        Args:
            task_id: "task1", "task2", or "task3"
            seed: optional integer for deterministic episodes

        Returns:
            Observation Pydantic model
        """
        client = self._get_client()
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed

        response = await client.post("/reset", json=payload)
        response.raise_for_status()
        return Observation.model_validate(response.json())

    async def step(
        self, action: dict[str, Any] | Action
    ) -> StepResult:
        """
        POST /step — apply one action.

        Args:
            action: Either an Action Pydantic model or a plain dict like:
                    {"action_type": "INVESTIGATE", "parameters": {"alert_id": "alert-001"}}

        Returns:
            StepResult Pydantic model
        """
        client = self._get_client()
        if isinstance(action, Action):
            payload = action.model_dump()
        else:
            payload = action

        response = await client.post("/step", json=payload)
        response.raise_for_status()
        return StepResult.model_validate(response.json())

    async def get_state(self) -> State:
        """
        GET /state — lightweight episode metadata.

        Returns:
            State Pydantic model
        """
        client = self._get_client()
        response = await client.get("/state")
        response.raise_for_status()
        return State.model_validate(response.json())

    async def health(self) -> dict[str, str]:
        """GET /health — confirm server is up."""
        client = self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Explicitly close the HTTP client. Call this if not using async with."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Sync wrappers (for non-async scripts) ────────────────────────────────
    def sync_reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        """Synchronous wrapper for reset(). Use in plain Python scripts."""
        return asyncio.run(self.reset(task_id, seed))

    def sync_step(self, action: dict[str, Any] | Action) -> StepResult:
        """Synchronous wrapper for step(). Use in plain Python scripts."""
        return asyncio.run(self.step(action))

    def sync_state(self) -> State:
        """Synchronous wrapper for get_state(). Use in plain Python scripts."""
        return asyncio.run(self.get_state())
