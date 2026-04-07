"""
models.py — IncidentMind Pydantic Data Contract
================================================
OWNER: Ritu
RULE: Finalise this file FIRST. Share field names with Amrita (for openenv.yaml)
      before writing any other file. No one else edits this file.

ACTION TYPES (7) — share with Amrita for openenv.yaml action_space.types:
    INVESTIGATE, MARK_ROOT_CAUSE, TRIGGER_RUNBOOK,
    GROUP_ALERTS, SUPPRESS_ALERT, QUERY_RUNBOOK, RESOLVE

OBSERVATION FIELDS (12) — share with Amrita for openenv.yaml observation_space.fields:
    alerts, investigated_alerts, alert_groups,
    root_cause_candidates, service_graph,
    triggered_runbooks, action_history, step_count,
    elapsed_seconds, task_id, max_steps, is_done
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# VALID ACTION TYPES  (single source of truth)
# ──────────────────────────────────────────────
ACTION_TYPES = Literal[
    "INVESTIGATE",
    "MARK_ROOT_CAUSE",
    "TRIGGER_RUNBOOK",
    "GROUP_ALERTS",
    "SUPPRESS_ALERT",
    "QUERY_RUNBOOK",
    "RESOLVE",
]


# ──────────────────────────────────────────────
# AlertModel
# ──────────────────────────────────────────────
class AlertModel(BaseModel):
    """
    Represents one alert visible to the agent.

    NOTE: Sneha's envs/alert_generator.py produces plain Python dataclasses.
    In server/environment.py you will convert each dataclass to this Pydantic model.
    The fields MUST match Sneha's dataclass field names exactly so the conversion
    is a simple dict(**vars(sneha_alert)) call.

    Hidden fields (is_noise, is_root_cause) are intentionally excluded here —
    the agent never sees them. The server keeps the raw Sneha objects internally.
    """

    id: str = Field(..., description="Unique alert identifier, e.g. 'alert-001'")
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Alert severity level"
    )
    source_service: str = Field(
        ..., description="Service name from ServiceGraph that fired this alert"
    )
    alert_type: str = Field(
        ..., description="Short code, e.g. 'DB_CONNECTION_POOL_EXHAUSTED'"
    )
    message: str = Field(..., description="Human-readable alert description")
    timestamp_offset: int = Field(
        ..., description="Seconds from episode start (for timeline reconstruction)"
    )

    model_config = {"frozen": True}  # Alerts are immutable once generated


# ──────────────────────────────────────────────
# ActionRecord
# ──────────────────────────────────────────────
class ActionRecord(BaseModel):
    """
    One entry in the Observation's action_history list.
    Records what the agent did, what happened, and what reward it earned.
    The Grader reads this list to compute partial scores.
    """

    action_type: ACTION_TYPES = Field(..., description="Which of the 7 actions was taken")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters: alert_id, alert_ids, runbook_id as needed",
    )
    outcome: str = Field(
        ..., description="Plain-text result of this action, e.g. 'alert-003 investigated'"
    )
    reward: float = Field(
        ..., description="Immediate reward this step earned (positive or negative)"
    )


# ──────────────────────────────────────────────
# Observation  (the 12 fields Amrita needs)
# ──────────────────────────────────────────────
class Observation(BaseModel):
    """
    Full environment state visible to the agent after every reset() or step().

    AMRITA: These 12 field names go verbatim into openenv.yaml observation_space.fields.
    Do NOT rename them without telling Amrita first.
    """

    # ── Alert state ──────────────────────────────────────────────────────────
    alerts: list[AlertModel] = Field(
        ..., description="All alerts currently visible to the agent"
    )
    investigated_alerts: list[str] = Field(
        default_factory=list,
        description="List of alert IDs the agent has already called INVESTIGATE on",
    )
    alert_groups: list[list[str]] = Field(
        default_factory=list,
        description="Groups of alert IDs the agent has clustered via GROUP_ALERTS",
    )

    # ── Root cause tracking ──────────────────────────────────────────────────
    root_cause_candidates: list[str] = Field(
        default_factory=list,
        description="Alert IDs the agent has marked via MARK_ROOT_CAUSE",
    )

    # ── Service topology ─────────────────────────────────────────────────────
    service_graph: dict[str, list[str]] = Field(
        ...,
        description="Adjacency list of the microservice DAG from Sneha's ServiceGraph",
    )

    # ── Remediation tracking ─────────────────────────────────────────────────
    triggered_runbooks: list[str] = Field(
        default_factory=list,
        description="List of runbook IDs the agent has triggered via TRIGGER_RUNBOOK",
    )

    # ── Episode progress ─────────────────────────────────────────────────────
    action_history: list[ActionRecord] = Field(
        default_factory=list,
        description="Ordered list of every action taken in this episode so far",
    )
    step_count: int = Field(
        0, description="Number of steps taken so far (0 at reset)"
    )
    elapsed_seconds: int = Field(
        0, description="Simulated wall-clock seconds elapsed in the episode"
    )

    # ── Task metadata ────────────────────────────────────────────────────────
    task_id: str = Field(
        ..., description="Which task is active: 'task1', 'task2', or 'task3'"
    )
    max_steps: int = Field(
        ..., description="Maximum steps allowed for this task (15 / 30 / 60)"
    )

    # ── Terminal flag ─────────────────────────────────────────────────────────
    is_done: bool = Field(
        False,
        description="True when RESOLVE is called or max_steps is reached",
    )


# ──────────────────────────────────────────────
# Action  (what the agent sends to /step)
# ──────────────────────────────────────────────
class Action(BaseModel):
    """
    The agent's action for one step. Sent as JSON body to POST /step.

    Parameter rules per action type:
        INVESTIGATE      → {"alert_id": "alert-001"}
        MARK_ROOT_CAUSE  → {"alert_id": "alert-001"}
        TRIGGER_RUNBOOK  → {"runbook_id": "RUNBOOK_DB_POOL_RESET"}
        GROUP_ALERTS     → {"alert_ids": ["alert-001", "alert-002"]}
        SUPPRESS_ALERT   → {"alert_id": "alert-001"}
        QUERY_RUNBOOK    → {"runbook_id": "RUNBOOK_CACHE_FAILOVER"}
        RESOLVE          → {}   (no parameters needed)

    AMRITA: The 7 action_type string values go into openenv.yaml action_space.types.
    """

    action_type: ACTION_TYPES = Field(
        ..., description="One of the 7 allowed action type strings"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters dictionary",
    )


# ──────────────────────────────────────────────
# State  (lightweight read from GET /state)
# ──────────────────────────────────────────────
class State(BaseModel):
    """
    Lightweight episode metadata. Returned by GET /state.
    Does NOT expose the full observation — use POST /reset or POST /step for that.
    """

    episode_id: str = Field(..., description="UUID of the current episode")
    step_count: int = Field(..., description="Steps taken in the current episode")
    elapsed_seconds: float = Field(
        ..., description="Simulated seconds elapsed in the episode"
    )
    task_id: str = Field(..., description="Active task identifier")


# ──────────────────────────────────────────────
# StepResult  (what /step returns)
# ──────────────────────────────────────────────
class StepResult(BaseModel):
    """
    Full result of one environment step. Returned by POST /step.

    When done=True and the episode ended via RESOLVE,
    info["grade_result"] will be a GradeResult dict.
    """

    observation: Observation = Field(
        ..., description="New environment state after applying the action"
    )
    reward: float = Field(
        ..., description="Immediate reward for this step (may be negative)"
    )
    done: bool = Field(
        ..., description="True if episode is over (RESOLVE called or max_steps hit)"
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra metadata. On terminal steps includes: "
            "'grade_result' (GradeResult dict), 'reason' (string)"
        ),
    )


# ──────────────────────────────────────────────
# ResetRequest  (body for POST /reset)
# ──────────────────────────────────────────────
class ResetRequest(BaseModel):
    """
    Request body for POST /reset.
    seed is optional — if omitted, a random seed is used.
    Using the same seed + task_id always produces the same episode (determinism).
    """

    task_id: Literal["task1", "task2", "task3"] = Field(
        ..., description="Which task to initialise"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility. None = random episode.",
    )


# ──────────────────────────────────────────────
# GradeResult  (returned inside StepResult.info)
# ──────────────────────────────────────────────
class GradeResult(BaseModel):
    """
    Deterministic score produced by Sneha's IncidentGrader.
    Placed in StepResult.info["grade_result"] when an episode ends.

    Scoring weights (for reference — Sneha implements the actual logic):
        Task 1: root_cause=0.40, runbook=0.35, resolve_called=0.15, efficiency=0.10
        Task 2: root_cause=0.35, runbook=0.30, noise_suppression=0.20,
                resolve_called=0.10, efficiency=0.05
        Task 3: root_cause=0.40, runbook=0.25, noise_suppression=0.15,
                alert_clustering=0.15, efficiency=0.05
    """

    total_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted total score between 0.0 and 1.0"
    )
    root_cause_score: float = Field(
        ..., ge=0.0, le=1.0, description="Partial score for root cause identification"
    )
    runbook_score: float = Field(
        ..., ge=0.0, le=1.0, description="Partial score for correct runbook usage"
    )
    noise_suppression_score: float = Field(
        ..., ge=0.0, le=1.0, description="Partial score for suppressing noise alerts"
    )
    efficiency_score: float = Field(
        ..., ge=0.0, le=1.0, description="Partial score for completing in fewer steps"
    )
    details: str = Field(
        ..., description="Plain-text explanation of what the agent did correctly/incorrectly"
    )
