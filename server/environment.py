"""
server/environment.py — IncidentMind Core Environment Orchestrator
===================================================================
OWNER: Ritu
IMPORTS FROM: All six of Sneha's envs/ modules
IMPORTED BY: server/app.py, demo.py

This file is the brain. It:
  1. Wraps Sneha's simulation modules into the OpenEnv reset/step/state contract
  2. Converts Sneha's plain Python dataclasses into Ritu's Pydantic models
  3. Applies dense reward shaping on every step
  4. Calls the Grader when the episode ends

WAIT: Do NOT write the import block until Sneha has pushed her branch and you have
merged it locally (git merge origin/feat/sneha-env-engine). The imports will fail
until Sneha's modules exist.

DENSE REWARD DESIGN (implement exactly as listed):
  +0.10  INVESTIGATE a new alert (first time)
  -0.02  INVESTIGATE an already-investigated alert (wasted step)
  +0.20  MARK_ROOT_CAUSE with a correct alert ID
  -0.15  MARK_ROOT_CAUSE with a wrong alert ID
  +0.25  TRIGGER_RUNBOOK with the correct runbook
  -0.20  TRIGGER_RUNBOOK with a wrong runbook
  +0.05  GROUP_ALERTS (any grouping — encourages clustering)
  +0.05  SUPPRESS_ALERT on a noise alert (correct suppression)
  -0.05  SUPPRESS_ALERT on a real alert (destroys evidence)
  +0.00  QUERY_RUNBOOK (free information action, no reward signal)
  +0.00  RESOLVE (actual score comes from Grader, not a step reward)

SIMULATED TIME: Each step adds 30 simulated seconds to elapsed_seconds.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Any, Optional

# ── Sneha's simulation modules ────────────────────────────────────────────────
from envs.service_graph import ServiceGraph
from envs.incident_generator import IncidentGenerator
from envs.alert_generator import AlertGenerator
from envs.grader import Grader
from envs.runbooks import RunbookRegistry
from envs.tasks import get_task

# (no module-level TASKS/RUNBOOKS globals: environment instance manages registry and task lookup)

# ── Ritu's own models ─────────────────────────────────────────────────────────
from models import (
    Action,
    ActionRecord,
    AlertModel,
    GradeResult,
    Observation,
    State,
    StepResult,
)

# ── Simulated time added per step (seconds) ───────────────────────────────────
SECONDS_PER_STEP = 30


# ─────────────────────────────────────────────────────────────────────────────
# Helper: convert Sneha's plain Alert dataclass → Ritu's AlertModel
# ─────────────────────────────────────────────────────────────────────────────
def _to_alert_model(sneha_alert: Any) -> AlertModel:
    """
    Sneha's AlertGenerator produces plain Python dataclasses.
    This converts them to AlertModel, stripping is_noise and is_root_cause
    so the agent cannot see the ground truth.

    If Sneha changes field names, update the mapping here — NOT in models.py.
    """
    return AlertModel(
        id=sneha_alert.id,
        severity=sneha_alert.severity,
        source_service=sneha_alert.source_service,
        alert_type=sneha_alert.alert_type,
        message=sneha_alert.message,
        timestamp_offset=int(round(sneha_alert.timestamp_offset)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# IncidentEnvironment
# ─────────────────────────────────────────────────────────────────────────────
class IncidentEnvironment:
    """
    Single shared environment instance. FastAPI's lifespan creates one instance
    at server startup and reuses it across requests. This means only one episode
    can be active at a time — which is correct for hackathon judging.

    Thread safety: FastAPI with uvicorn (single worker) is single-threaded by
    default for async handlers. Do not add threading without understanding the
    implications.
    """

    def __init__(self) -> None:
        # ── Sneha's stateless module instances ───────────────────────────────
        self._service_graph = ServiceGraph()
        self._incident_generator = IncidentGenerator()
        self._alert_generator = AlertGenerator()
        self._grader = Grader()
        self._runbook_registry = RunbookRegistry()

        # ── Episode state (reset on every reset() call) ───────────────────────
        self._episode_id: str = ""
        self._task_id: str = ""
        self._task_id_int: Optional[int] = None
        self._task_config: Any = None          # Sneha's TaskConfig object
        self._ground_truth: Any = None         # Sneha's IncidentScenario (hidden)
        self._raw_alerts: list[Any] = []       # Sneha's Alert dataclasses (hidden)
        self._step_count: int = 0
        self._elapsed_seconds: int = 0
        self._action_history: list[ActionRecord] = []
        self._triggered_runbooks: list[str] = []
        self._investigated_alerts: list[str] = []
        self._alert_groups: list[list[str]] = []
        self._root_cause_candidates: list[str] = []
        self._suppressed_alert_ids: set[str] = set()
        self._is_done: bool = False

    def _parse_task_id(self, task_id: str) -> int:
        """Convert task string (task1/task2/task3) to numeric task id."""
        if not task_id.startswith("task"):
            raise ValueError(f"Invalid task_id: {task_id}. Expected 'task1', 'task2', or 'task3'.")

        try:
            task_int = int(task_id.replace("task", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid task_id: {task_id}. "
                             "Expected 'task1', 'task2', or 'task3'.") from exc

        if task_int not in (1, 2, 3):
            raise ValueError(f"Invalid task_id: {task_id}. Must be 'task1', 'task2', or 'task3'.")

        return task_int

    # ──────────────────────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        """
        Start a new episode. Called by POST /reset.

        Steps (follow the document order exactly):
          1. Generate a fresh episode UUID
          2. Load task config from Sneha's tasks module
          3. Generate hidden ground truth (IncidentScenario)
          4. Generate the alert stream from that ground truth
          5. Get the service graph adjacency list
          6. Zero out all mutable episode state
          7. Return a full Observation
        """
        # Step 1 — New episode
        self._episode_id = str(uuid.uuid4())
        self._task_id = task_id

        # Step 2 — Load task configuration
        self._task_id_int = self._parse_task_id(task_id)
        self._task_config = get_task(self._task_id_int)

        # Step 3 — Generate hidden ground truth (agent never sees this)
        self._incident_generator = IncidentGenerator(seed=seed or 42)
        self._ground_truth = self._incident_generator.generate(
            task_id=self._task_id_int,
        )

        # Step 4 — Generate the alert stream from this ground truth
        self._alert_generator = AlertGenerator(seed=seed or 42)
        self._raw_alerts = self._alert_generator.generate(self._ground_truth)

        # Step 5 — Get service graph as adjacency list dict
        graph = self._service_graph.get_graph()
        service_graph_dict: dict[str, list[str]] = {
            node: list(graph.successors(node)) for node in graph.nodes()
        }

        # Step 6 — Zero out all episode state
        self._step_count = 0
        self._elapsed_seconds = 0
        self._action_history = []
        self._triggered_runbooks = []
        self._investigated_alerts = []
        self._alert_groups = []
        self._root_cause_candidates = []
        self._suppressed_alert_ids = set()
        self._is_done = False

        # Step 7 — Build and return the Observation
        return self._build_observation(service_graph_dict)

    # ──────────────────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: Action) -> StepResult:
        """
        Apply one agent action and return the result. Called by POST /step.

        Returns HTTP 422-compatible error in info if action_type is invalid
        (FastAPI handles the 422 for bad Pydantic validation; this handles
        valid action_type strings with missing/wrong parameters).
        """
        if self._is_done:
            # Episode already over — agent called step() after RESOLVE or max_steps
            return StepResult(
                observation=self._build_observation(self._service_graph.get_graph()),
                reward=0.0,
                done=True,
                info={"error": "Episode is already done. Call /reset to start a new episode."},
            )

        reward: float = 0.0
        outcome: str = ""

        # ── Dispatch to action handler ────────────────────────────────────────
        action_type = action.action_type
        params = action.parameters

        if action_type == "INVESTIGATE":
            reward, outcome = self._handle_investigate(params)

        elif action_type == "MARK_ROOT_CAUSE":
            reward, outcome = self._handle_mark_root_cause(params)

        elif action_type == "TRIGGER_RUNBOOK":
            reward, outcome = self._handle_trigger_runbook(params)

        elif action_type == "GROUP_ALERTS":
            reward, outcome = self._handle_group_alerts(params)

        elif action_type == "SUPPRESS_ALERT":
            reward, outcome = self._handle_suppress_alert(params)

        elif action_type == "QUERY_RUNBOOK":
            reward, outcome = self._handle_query_runbook(params)

        elif action_type == "RESOLVE":
            reward, outcome = 0.0, "Agent called RESOLVE — grading episode."

        # ── Update episode counters ───────────────────────────────────────────
        self._step_count += 1
        self._elapsed_seconds += SECONDS_PER_STEP

        # ── Record the action in history ──────────────────────────────────────
        record = ActionRecord(
            action_type=action_type,
            parameters=params,
            outcome=outcome,
            reward=reward,
        )
        self._action_history.append(record)

        # ── Check terminal conditions ─────────────────────────────────────────
        done = False
        info: dict[str, Any] = {}

        if action_type == "RESOLVE" or self._step_count >= self._task_config.max_steps:
            done = True
            self._is_done = True

            if action_type != "RESOLVE":
                info["reason"] = "max_steps_reached"
            else:
                info["reason"] = "agent_resolved"

            # Call Sneha's grader — this is the authoritative final score
            # Convert internal history to grader's expected action schema.
            def _map_action_type(at: str) -> str:
                return {
                    "INVESTIGATE": "INVESTIGATE",
                    "MARK_ROOT_CAUSE": "IDENTIFY_ROOT_CAUSE",
                    "TRIGGER_RUNBOOK": "APPLY_RUNBOOK",
                    "GROUP_ALERTS": "GROUP_ALERTS",
                    "SUPPRESS_ALERT": "DISMISS_NOISE",
                    "QUERY_RUNBOOK": "QUERY_RUNBOOK",
                    "RESOLVE": "RESOLVE",
                }.get(at, at)

            agent_actions = []
            for idx, rec in enumerate(self._action_history):
                rec_dict = rec.model_dump() if hasattr(rec, "model_dump") else rec.dict()
                params = rec_dict.get("parameters", {}) or {}
                action_payload = {
                    "type": _map_action_type(rec_dict.get("action_type", "")),
                    "step": idx,
                    **params,
                }
                agent_actions.append(action_payload)

            sneha_grade = self._grader.grade(
                ground_truth=asdict(self._ground_truth),
                agent_actions=agent_actions,
                task_id=self._task_id_int or self._parse_task_id(self._task_id),
            )

            # Convert Sneha's GradeResult (plain Python) → Ritu's Pydantic GradeResult
            import json as _json
            grade_result = GradeResult(
                total_score=sneha_grade.total_score,
                root_cause_score=sneha_grade.root_cause_score,
                runbook_score=sneha_grade.runbook_score,
                noise_suppression_score=sneha_grade.noise_suppression_score,
                efficiency_score=sneha_grade.efficiency_score,
                details=_json.dumps(sneha_grade.details) if isinstance(sneha_grade.details, dict) else str(sneha_grade.details),
            )
            info["grade_result"] = grade_result.model_dump()

        # ── Build new observation ─────────────────────────────────────────────
        graph = self._service_graph.get_graph()
        service_graph_dict = {node: list(graph.successors(node)) for node in graph.nodes()}

        obs = self._build_observation(
            service_graph=service_graph_dict,
            is_done=done,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # state()
    # ──────────────────────────────────────────────────────────────────────────
    def state(self) -> State:
        """
        Pure read — returns lightweight episode metadata.
        MUST NOT change any internal state.
        Called by GET /state.
        """
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            elapsed_seconds=float(self._elapsed_seconds),
            task_id=self._task_id,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Action handlers (private)
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_investigate(self, params: dict) -> tuple[float, str]:
        alert_id = params.get("alert_id", "")
        if not alert_id:
            return -0.05, "INVESTIGATE failed: 'alert_id' parameter is missing."

        if alert_id in self._investigated_alerts:
            return -0.02, f"INVESTIGATE: alert {alert_id} was already investigated (wasted step)."

        # Check alert exists in visible alert stream
        known_ids = {a.id for a in self._raw_alerts if a.id not in self._suppressed_alert_ids}
        if alert_id not in known_ids:
            return -0.05, f"INVESTIGATE: alert {alert_id} not found in active alerts."

        self._investigated_alerts.append(alert_id)
        return 0.10, f"INVESTIGATE: alert {alert_id} investigated successfully."

    def _handle_mark_root_cause(self, params: dict) -> tuple[float, str]:
        alert_id = params.get("alert_id", "")
        if not alert_id:
            return -0.05, "MARK_ROOT_CAUSE failed: 'alert_id' parameter is missing."

        # Check against ground truth (agent doesn't see this, but the env does)
        is_correct = alert_id in self._ground_truth.root_cause_alert_ids

        if alert_id not in self._root_cause_candidates:
            self._root_cause_candidates.append(alert_id)

        if is_correct:
            return 0.20, f"MARK_ROOT_CAUSE: {alert_id} is a correct root cause. ✓"
        else:
            return -0.15, f"MARK_ROOT_CAUSE: {alert_id} is NOT a root cause. ✗"

    def _handle_trigger_runbook(self, params: dict) -> tuple[float, str]:
        runbook_id = params.get("runbook_id", "")
        if not runbook_id:
            return -0.05, "TRIGGER_RUNBOOK failed: 'runbook_id' parameter is missing."

        # Check runbook exists
        try:
            self._runbook_registry.get(runbook_id)
        except KeyError:
            return -0.10, f"TRIGGER_RUNBOOK: runbook '{runbook_id}' does not exist."

        is_correct = runbook_id in getattr(self._ground_truth, "correct_runbook_ids", [])

        if runbook_id not in self._triggered_runbooks:
            self._triggered_runbooks.append(runbook_id)

        if is_correct:
            return 0.25, f"TRIGGER_RUNBOOK: {runbook_id} is the correct runbook. ✓"
        else:
            return -0.20, f"TRIGGER_RUNBOOK: {runbook_id} is the wrong runbook. ✗"

    def _handle_group_alerts(self, params: dict) -> tuple[float, str]:
        alert_ids = params.get("alert_ids", [])
        if not isinstance(alert_ids, list) or len(alert_ids) < 2:
            return -0.05, "GROUP_ALERTS failed: 'alert_ids' must be a list of at least 2 alert IDs."

        self._alert_groups.append(alert_ids)
        return 0.05, f"GROUP_ALERTS: grouped {len(alert_ids)} alerts into cluster {len(self._alert_groups)}."

    def _handle_suppress_alert(self, params: dict) -> tuple[float, str]:
        alert_id = params.get("alert_id", "")
        if not alert_id:
            return -0.05, "SUPPRESS_ALERT failed: 'alert_id' parameter is missing."

        # Check if this is actually a noise alert (agent gets reward for correct suppression)
        raw = next((a for a in self._raw_alerts if a.id == alert_id), None)
        if raw is None:
            return -0.05, f"SUPPRESS_ALERT: alert {alert_id} not found."

        self._suppressed_alert_ids.add(alert_id)

        if raw.is_noise:
            return 0.05, f"SUPPRESS_ALERT: {alert_id} was noise. Correctly suppressed. ✓"
        else:
            return -0.05, f"SUPPRESS_ALERT: {alert_id} was a real alert. Suppression harmful. ✗"

    def _handle_query_runbook(self, params: dict) -> tuple[float, str]:
        runbook_id = params.get("runbook_id", "")
        if not runbook_id:
            return 0.00, f"QUERY_RUNBOOK: runbook '{runbook_id}' not found in catalogue."

        try:
            runbook = self._runbook_registry.get(runbook_id)
        except KeyError:
            return 0.00, f"QUERY_RUNBOOK: runbook '{runbook_id}' not found in catalogue."

        desc = getattr(runbook, "description", str(runbook))
        return 0.00, f"QUERY_RUNBOOK: {runbook_id} — {desc}"

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _build_observation(
        self,
        service_graph: dict[str, list[str]],
        is_done: bool = False,
    ) -> Observation:
        """
        Build a full Observation from current internal state.
        Converts Sneha's raw Alert dataclasses → AlertModel, filtering suppressed ones.
        """
        visible_alerts = [
            _to_alert_model(a)
            for a in self._raw_alerts
            if a.id not in self._suppressed_alert_ids
        ]

        return Observation(
            alerts=visible_alerts,
            investigated_alerts=list(self._investigated_alerts),
            alert_groups=list(self._alert_groups),
            root_cause_candidates=list(self._root_cause_candidates),
            service_graph=service_graph,
            triggered_runbooks=list(self._triggered_runbooks),
            action_history=list(self._action_history),
            step_count=self._step_count,
            elapsed_seconds=self._elapsed_seconds,
            task_id=self._task_id,
            max_steps=self._task_config.max_steps if self._task_config else 0,
            is_done=is_done,
        )
