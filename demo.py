"""
demo.py — IncidentMind Local Smoke Test (No Docker, No HTTP)
============================================================
OWNER: Ritu
PURPOSE: Validate that Sneha's envs + Ritu's environment.py integrate correctly
         BEFORE Amrita builds the Docker container.

HOW TO RUN:
    # From project root, after merging Sneha's branch:
    python demo.py

WHAT THIS TESTS:
  - All six of Sneha's envs modules import without errors
  - IncidentEnvironment.reset() returns a valid Observation
  - IncidentEnvironment.step() applies actions and returns correct structure
  - Grader fires and returns a real score on RESOLVE
  - All three tasks work end-to-end

SHARE THE OUTPUT with Sneha so she can verify the grader is scoring correctly.
If you see total_score = 0.0 for every task, the grader has a bug.
"""

from __future__ import annotations

import json
from typing import Any

# These imports will fail until Sneha's branch is merged locally.
# Run: git fetch origin && git merge origin/feat/sneha-env-engine
from server.environment import IncidentEnvironment
from models import Observation, StepResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_observation(obs: Observation, label: str = "") -> None:
    if label:
        print(f"\n{'─'*50}")
        print(f"  {label}")
        print(f"{'─'*50}")
    print(f"  Task:       {obs.task_id}")
    print(f"  Step:       {obs.step_count}/{obs.max_steps}")
    print(f"  Alerts:     {len(obs.alerts)} visible")
    print(f"  Services:   {len(obs.service_graph)} nodes in graph")
    print(f"  Is done:    {obs.is_done}")
    print()
    if obs.alerts:
        print("  Alert snapshot:")
        for a in obs.alerts[:5]:  # Show first 5
            print(f"    [{a.severity}] {a.id} | {a.source_service} | {a.alert_type}")
        if len(obs.alerts) > 5:
            print(f"    ... and {len(obs.alerts) - 5} more")


def print_step_result(result: StepResult, step_num: int) -> None:
    print(f"  Step {step_num:>2}: {result.observation.action_history[-1].action_type if result.observation.action_history else '?':<20}"
          f" reward={result.reward:+.3f}  done={result.done}")
    if result.info.get("error"):
        print(f"         ERROR: {result.info['error']}")


def print_grade(grade: dict[str, Any], task_id: str) -> None:
    print(f"\n  ┌─ Final Grade: {task_id} ─────────────────────────────")
    print(f"  │  Total Score:            {grade['total_score']:.4f}")
    print(f"  │  Root Cause Score:       {grade['root_cause_score']:.4f}")
    print(f"  │  Runbook Score:          {grade['runbook_score']:.4f}")
    print(f"  │  Noise Suppression:      {grade['noise_suppression_score']:.4f}")
    print(f"  │  Efficiency Score:       {grade['efficiency_score']:.4f}")
    print(f"  │  Details: {grade['details'][:80]}")
    print(f"  └──────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy (payment-db connection pool exhaustion)
# ─────────────────────────────────────────────────────────────────────────────
def run_task1(env: IncidentEnvironment) -> None:
    print("\n" + "="*60)
    print("  TASK 1 — EASY (seed=42, max_steps=15)")
    print("="*60)

    obs = env.reset(task_id="task1", seed=42)
    print_observation(obs, "Initial Observation")

    # Hardcoded action sequence for Task 1
    # Real root cause: payment-db DB_CONNECTION_POOL_EXHAUSTED
    # Correct runbook: RUNBOOK_DB_POOL_RESET
    actions = [
        {"action_type": "INVESTIGATE",       "parameters": {"alert_id": obs.alerts[0].id}},
        {"action_type": "INVESTIGATE",       "parameters": {"alert_id": obs.alerts[1].id}},
        {"action_type": "MARK_ROOT_CAUSE",   "parameters": {"alert_id": obs.alerts[0].id}},
        {"action_type": "TRIGGER_RUNBOOK",   "parameters": {"runbook_id": "rb_db_failover"}},
        {"action_type": "RESOLVE",           "parameters": {}},
    ]

    from models import Action
    for i, action_dict in enumerate(actions, start=1):
        action = Action(**action_dict)
        result = env.step(action)
        print_step_result(result, i)
        if result.done:
            grade = result.info.get("grade_result", {})
            print_grade(grade, "task1")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium (redis-primary timeout cascading to api-gateway)
# ─────────────────────────────────────────────────────────────────────────────
def run_task2(env: IncidentEnvironment) -> None:
    print("\n" + "="*60)
    print("  TASK 2 — MEDIUM (seed=42, max_steps=30)")
    print("="*60)

    obs = env.reset(task_id="task2", seed=42)
    print_observation(obs, "Initial Observation")

    # Real root cause: redis-primary CACHE_CONNECTION_TIMEOUT
    # Correct runbook: RUNBOOK_CACHE_FAILOVER
    from models import Action
    actions = [
        {"action_type": "INVESTIGATE",     "parameters": {"alert_id": obs.alerts[0].id}},
        {"action_type": "INVESTIGATE",     "parameters": {"alert_id": obs.alerts[1].id}},
        {"action_type": "INVESTIGATE",     "parameters": {"alert_id": obs.alerts[2].id}},
        {"action_type": "GROUP_ALERTS",    "parameters": {"alert_ids": [a.id for a in obs.alerts[:3]]}},
        {"action_type": "MARK_ROOT_CAUSE", "parameters": {"alert_id": obs.alerts[0].id}},
        {"action_type": "TRIGGER_RUNBOOK", "parameters": {"runbook_id": "rb_cache_flush_restart"}},
        {"action_type": "RESOLVE",         "parameters": {}},
    ]

    for i, action_dict in enumerate(actions, start=1):
        action = Action(**action_dict)
        result = env.step(action)
        print_step_result(result, i)
        if result.done:
            grade = result.info.get("grade_result", {})
            print_grade(grade, "task2")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard (dual root cause: DISK_IO_ERROR + MEMORY_LIMIT_EXCEEDED)
# ─────────────────────────────────────────────────────────────────────────────
def run_task3(env: IncidentEnvironment) -> None:
    print("\n" + "="*60)
    print("  TASK 3 — HARD (seed=42, max_steps=60)")
    print("="*60)

    obs = env.reset(task_id="task3", seed=42)
    print_observation(obs, "Initial Observation")

    # Real root causes: storage-node-3 DISK_IO_ERROR + ml-inference MEMORY_LIMIT_EXCEEDED
    # Correct runbooks: RUNBOOK_DISK_REPAIR + RUNBOOK_ML_SERVICE_RESTART
    from models import Action
    actions = [
        {"action_type": "INVESTIGATE",     "parameters": {"alert_id": obs.alerts[0].id}},
        {"action_type": "INVESTIGATE",     "parameters": {"alert_id": obs.alerts[1].id}},
        {"action_type": "INVESTIGATE",     "parameters": {"alert_id": obs.alerts[2].id}},
        {"action_type": "GROUP_ALERTS",    "parameters": {"alert_ids": [a.id for a in obs.alerts[:4]]}},
        # Suppress some noise (Task 3 has 8-12 noise alerts)
        {"action_type": "SUPPRESS_ALERT",  "parameters": {"alert_id": obs.alerts[-1].id}},
        {"action_type": "MARK_ROOT_CAUSE", "parameters": {"alert_id": obs.alerts[0].id}},
        {"action_type": "MARK_ROOT_CAUSE", "parameters": {"alert_id": obs.alerts[1].id}},
        {"action_type": "TRIGGER_RUNBOOK", "parameters": {"runbook_id": "rb_storage_volume_remount"}},
        {"action_type": "TRIGGER_RUNBOOK", "parameters": {"runbook_id": "rb_ml_model_rollback"}},
        {"action_type": "RESOLVE",         "parameters": {}},
    ]

    for i, action_dict in enumerate(actions, start=1):
        action = Action(**action_dict)
        result = env.step(action)
        print_step_result(result, i)
        if result.done:
            grade = result.info.get("grade_result", {})
            print_grade(grade, "task3")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("IncidentMind — Local Smoke Test")
    print("(No Docker, No HTTP — direct Python import)")
    print()

    env = IncidentEnvironment()

    run_task1(env)
    run_task2(env)
    run_task3(env)

    print("\n" + "="*60)
    print("  Smoke test complete.")
    print("  Share this output with Sneha to verify grader correctness.")
    print("="*60)


if __name__ == "__main__":
    main()
