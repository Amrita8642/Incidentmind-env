# Minor update by Sneha
"""
tasks.py
========
Task definitions for the IncidentMind RL environment.

Three tasks of increasing difficulty:
  Task 1 — Beginner  : single root cause, 5–8 alerts, no noise
  Task 2 — Moderate  : cascade chain, 15–25 alerts, moderate noise
  Task 3 — Expert    : dual root causes, 35–50 alerts, heavy noise + red herrings

Each task specifies constraints used by:
  - AlertGenerator (to generate the right number of alerts)
  - Grader (for passing_score and max_steps)
  - Environment (for episode configuration)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskDefinition:
    """
    Immutable task configuration.

    alert_count_range  : (min, max) total alerts in the episode
    noise_percentage   : fraction of alerts that are noise (0.0 = none)
    red_herring_count  : number of convincing-but-wrong high-sev alerts
    """
    task_id: int
    name: str
    difficulty: str                         # "beginner" | "moderate" | "expert"
    max_steps: int                          # maximum agent steps per episode
    passing_score: float                    # minimum total_score to pass
    alert_count_range: Tuple[int, int]      # (min_alerts, max_alerts)
    noise_percentage: float                 # 0.0 – 1.0
    red_herring_count: int
    description: str
    hint: str = ""                          # optional RL hint for curriculum learning
    reward_shaping: Dict[str, float] = field(default_factory=dict)

    def validate_alert_count(self, count: int) -> bool:
        """Check that generated alert count falls within task constraints."""
        lo, hi = self.alert_count_range
        return lo <= count <= hi

    def expected_noise_count(self, total_alerts: int) -> int:
        """Expected number of noise alerts given total alert count."""
        return int(round(total_alerts * self.noise_percentage))


# ---------------------------------------------------------------------------
# Task 1 — Single root cause, no noise
# ---------------------------------------------------------------------------

TASK1 = TaskDefinition(
    task_id=1,
    name="Payment DB Failure",
    difficulty="beginner",
    max_steps=20,
    passing_score=0.70,
    alert_count_range=(5, 8),
    noise_percentage=0.0,
    red_herring_count=0,
    description=(
        "A single critical service (payment-db) has failed. "
        "Alerts are clean and cascade is shallow. "
        "Agent must identify payment-db as root cause and apply correct remediation runbook. "
        "No noise — every alert is signal."
    ),
    hint="All alerts are meaningful. Look for the CRITICAL alert at T=0.",
    reward_shaping={
        "correct_root_cause_bonus":   +0.30,
        "correct_runbook_bonus":       +0.20,
        "premature_resolve_penalty":  -0.25,
        "greedy_penalty":             -0.30,
    },
)

# ---------------------------------------------------------------------------
# Task 2 — Cascading failure, moderate noise
# ---------------------------------------------------------------------------

TASK2 = TaskDefinition(
    task_id=2,
    name="Redis Cache Cascade: Auth → API → Frontend",
    difficulty="moderate",
    max_steps=40,
    passing_score=0.65,
    alert_count_range=(15, 25),
    noise_percentage=0.25,     # ~25% of alerts are noise
    red_herring_count=1,
    description=(
        "A redis-cache failure cascades through auth-service → api-gateway → frontend-web. "
        "The cascade unfolds over 30–60 simulated seconds. "
        "Agent must distinguish root cause from cascade effects and ignore noise. "
        "One convincing red herring (metrics-collector) fires early and misleads greedy agents."
    ),
    hint=(
        "The CRITICAL alert at T=0 may not be the root cause if it's from a non-data-tier service. "
        "Check dependency context before marking root cause."
    ),
    reward_shaping={
        "correct_root_cause_bonus":   +0.25,
        "correct_runbook_sequence":   +0.15,
        "noise_dismiss_bonus":        +0.10,
        "red_herring_trap_penalty":   -0.20,
        "greedy_penalty":             -0.30,
    },
)

# ---------------------------------------------------------------------------
# Task 3 — Dual root causes, heavy noise and red herrings
# ---------------------------------------------------------------------------

TASK3 = TaskDefinition(
    task_id=3,
    name="Dual Root Cause: Storage + ML Inference Simultaneous Failure",
    difficulty="expert",
    max_steps=75,
    passing_score=0.60,
    alert_count_range=(35, 50),
    noise_percentage=0.40,     # ~40% of alerts are noise
    red_herring_count=3,
    description=(
        "Two independent failures (storage-node and ml-inference) occur within 20 seconds of each other, "
        "producing interleaved cascades affecting order-service, order-db, and api-gateway. "
        "Heavy noise (40%) and 3 convincing red herrings (message-queue, notification-svc, frontend-web). "
        "Agent must identify BOTH root causes and apply the correct pair of runbooks. "
        "This is a multi-causal incident — single-root-cause reasoning will fail."
    ),
    hint=(
        "Look for two independent CRITICAL alerts whose services have no dependency between them. "
        "Two separate cascade chains will be visible if you trace dependency_context carefully."
    ),
    reward_shaping={
        "both_root_causes_bonus":     +0.30,
        "partial_root_cause":         +0.15,    # credit for finding one of two
        "correct_runbook_pair":       +0.20,
        "noise_dismiss_bonus":        +0.10,
        "red_herring_trap_penalty":   -0.15,    # each red herring marked costs this
        "greedy_penalty":             -0.30,
        "missed_second_root_cause":   -0.20,
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[int, TaskDefinition] = {
    1: TASK1,
    2: TASK2,
    3: TASK3,
}


def get_task(task_id: int) -> TaskDefinition:
    """
    Retrieve a TaskDefinition by ID.

    Raises:
        ValueError: if task_id is not 1, 2, or 3.
    """
    if task_id not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id: {task_id}. "
            f"Available tasks: {sorted(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[task_id]


def list_tasks() -> List[TaskDefinition]:
    """Return all task definitions in order."""
    return [_TASK_REGISTRY[k] for k in sorted(_TASK_REGISTRY)]
