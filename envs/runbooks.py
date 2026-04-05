"""
runbooks.py
===========
Defines 7 SRE runbooks for the IncidentMind simulation.

Each runbook has:
  - id, description
  - trigger_condition(ground_truth) → bool   — determines if runbook is appropriate
  - effect(simulated_state) → SimulatedState  — mutates system state on execution

Effects are realistic:
  - correct runbooks restore services, stop cascade propagation, resolve alerts
  - wrong runbooks may make state worse (or do nothing) — tested by grader
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Simulated system state — passed into and returned from runbook effects
# ---------------------------------------------------------------------------

@dataclass
class SimulatedState:
    """
    Mutable snapshot of the system state during an incident.

    The grader passes this into runbook.effect() and reads back the result.
    """
    service_health: Dict[str, float]                # service_name → health_score
    resolved_alerts: Set[str] = field(default_factory=set)   # alert IDs resolved
    stopped_cascades: Set[str] = field(default_factory=set)  # services no longer cascading
    side_effects: List[str] = field(default_factory=list)    # log of applied effects

    def apply_health(self, service: str, delta: float) -> None:
        import numpy as np
        current = self.service_health.get(service, 1.0)
        self.service_health[service] = float(np.clip(current + delta, 0.0, 1.0))

    def resolve_alerts_for(self, service: str, all_alerts_by_service: Dict[str, List[str]]) -> None:
        """Mark all alerts originating from *service* as resolved."""
        for alert_id in all_alerts_by_service.get(service, []):
            self.resolved_alerts.add(alert_id)

    def stop_cascade_from(self, service: str) -> None:
        self.stopped_cascades.add(service)
        self.side_effects.append(f"cascade_stopped:{service}")

    def log(self, msg: str) -> None:
        self.side_effects.append(msg)


# ---------------------------------------------------------------------------
# Runbook dataclass
# ---------------------------------------------------------------------------

@dataclass
class Runbook:
    """
    An SRE runbook that can be applied by the agent.

    trigger_condition:
        fn(ground_truth: dict) → bool
        Returns True if this runbook is *appropriate* for the given incident.
        ground_truth contains: root_services, task_id, involved_services.

    effect:
        fn(state: SimulatedState, ground_truth: dict) → SimulatedState
        Mutates and returns the SimulatedState after applying the runbook.
    """
    id: str
    name: str
    description: str
    applicable_services: List[str]
    trigger_condition: Callable[[Dict], bool]
    effect: Callable[[SimulatedState, Dict], SimulatedState]

    def is_applicable(self, ground_truth: Dict) -> bool:
        return self.trigger_condition(ground_truth)

    def apply(self, state: SimulatedState, ground_truth: Dict) -> SimulatedState:
        return self.effect(state, ground_truth)


# ---------------------------------------------------------------------------
# Helper: build alert index by service (used inside effect fns)
# ---------------------------------------------------------------------------

def _alerts_by_service(ground_truth: Dict) -> Dict[str, List[str]]:
    """Extract alert_id groupings per service from ground_truth."""
    return ground_truth.get("alerts_by_service", {})


# ---------------------------------------------------------------------------
# Runbook definitions
# ---------------------------------------------------------------------------

def _rb_db_failover_condition(gt: Dict) -> bool:
    root_svcs = gt.get("root_services", [gt.get("root_service", "")])
    if isinstance(root_svcs, str):
        root_svcs = [root_svcs]
    return "payment-db" in root_svcs

def _rb_db_failover_effect(state: SimulatedState, gt: Dict) -> SimulatedState:
    state.apply_health("payment-db", +0.90)
    state.stop_cascade_from("payment-db")
    state.resolve_alerts_for("payment-db", _alerts_by_service(gt))
    state.log("payment-db failover completed — replica promoted to primary")
    state.log("payment-service connection pool refreshed")
    return state

RB_DB_FAILOVER = Runbook(
    id="rb_db_failover",
    name="Database Primary Failover",
    description=(
        "Triggers manual failover of payment-db to standby replica. "
        "Promotes standby, updates DNS CNAME, drains in-flight transactions, "
        "and flushes connection pool on payment-service. "
        "Required when primary is unreachable or replication lag > 30s."
    ),
    applicable_services=["payment-db", "payment-service"],
    trigger_condition=_rb_db_failover_condition,
    effect=_rb_db_failover_effect,
)

# ------------------------------------------------------------------

def _rb_service_restart_condition(gt: Dict) -> bool:
    # Applicable whenever there are cascading services
    return len(gt.get("involved_services", [])) > 1

def _rb_service_restart_effect(state: SimulatedState, gt: Dict) -> SimulatedState:
    # Restarts all involved services except DBs and infra
    non_db = [s for s in gt.get("involved_services", [])
              if "db" not in s and s not in ("metrics-collector",)]
    for svc in non_db:
        state.apply_health(svc, +0.65)
        state.stop_cascade_from(svc)
        state.resolve_alerts_for(svc, _alerts_by_service(gt))
    state.log(f"graceful restart applied to: {non_db}")
    return state

RB_SERVICE_RESTART = Runbook(
    id="rb_service_restart",
    name="Graceful Service Restart",
    description=(
        "Performs rolling restart of affected application-tier services. "
        "Drains current connections, waits for inflight requests to complete, "
        "then restarts pods / processes. Clears in-memory corruption states. "
        "Not appropriate as first action — diagnose root cause first."
    ),
    applicable_services=["*"],
    trigger_condition=_rb_service_restart_condition,
    effect=_rb_service_restart_effect,
)

# ------------------------------------------------------------------

def _rb_cache_flush_condition(gt: Dict) -> bool:
    root_svcs = gt.get("root_services", [gt.get("root_service", "")])
    if isinstance(root_svcs, str):
        root_svcs = [root_svcs]
    return "redis-cache" in root_svcs

def _rb_cache_flush_effect(state: SimulatedState, gt: Dict) -> SimulatedState:
    state.apply_health("redis-cache", +0.95)
    state.stop_cascade_from("redis-cache")
    state.resolve_alerts_for("redis-cache", _alerts_by_service(gt))
    # Downstream services recover partially once cache is back
    for svc in ("auth-service", "payment-service"):
        state.apply_health(svc, +0.50)
    state.log("redis-cache flushed and restarted — cluster re-sharding complete")
    state.log("auth-service and payment-service session cache warming")
    return state

RB_CACHE_FLUSH_RESTART = Runbook(
    id="rb_cache_flush_restart",
    name="Redis Cache Flush and Cluster Restart",
    description=(
        "Flushes all redis-cache data, restarts cluster nodes in order, "
        "waits for cluster MEET and slot assignment, then triggers session "
        "cache warm-up on auth-service. "
        "Use when redis OOM, cluster partition, or key eviction storm detected."
    ),
    applicable_services=["redis-cache", "auth-service", "payment-service"],
    trigger_condition=_rb_cache_flush_condition,
    effect=_rb_cache_flush_effect,
)

# ------------------------------------------------------------------

def _rb_auth_token_condition(gt: Dict) -> bool:
    return "auth-service" in gt.get("involved_services", [])

def _rb_auth_token_effect(state: SimulatedState, gt: Dict) -> SimulatedState:
    state.apply_health("auth-service", +0.80)
    state.stop_cascade_from("auth-service")
    state.resolve_alerts_for("auth-service", _alerts_by_service(gt))
    state.log("auth-service JWT signing key rotated — old tokens invalidated")
    state.log("session store rebuild triggered — users will re-authenticate")
    return state

RB_AUTH_TOKEN_INVALIDATE = Runbook(
    id="rb_auth_token_invalidate",
    name="Auth Token Invalidation and Session Rebuild",
    description=(
        "Force-invalidates all active JWT tokens and rebuilds the session store "
        "from persistent backend. Rotates signing keys. "
        "Use when auth-service is in a bad state due to corrupted cache or "
        "stale session data causing validation loops."
    ),
    applicable_services=["auth-service"],
    trigger_condition=_rb_auth_token_condition,
    effect=_rb_auth_token_effect,
)

# ------------------------------------------------------------------

def _rb_storage_remount_condition(gt: Dict) -> bool:
    root_svcs = gt.get("root_services", [gt.get("root_service", "")])
    if isinstance(root_svcs, str):
        root_svcs = [root_svcs]
    return "storage-node" in root_svcs

def _rb_storage_remount_effect(state: SimulatedState, gt: Dict) -> SimulatedState:
    state.apply_health("storage-node", +0.95)
    state.stop_cascade_from("storage-node")
    state.resolve_alerts_for("storage-node", _alerts_by_service(gt))
    # Downstream unblocked
    for svc in ("order-service", "ml-inference"):
        state.apply_health(svc, +0.60)
    state.log("storage-node volume remounted read-write — fsck passed")
    state.log("order-service and ml-inference storage handles refreshed")
    return state

RB_STORAGE_VOLUME_REMOUNT = Runbook(
    id="rb_storage_volume_remount",
    name="Storage Volume Force Remount",
    description=(
        "Unmounts and force-remounts the storage-node data volume. "
        "Runs fsck to check filesystem integrity. Reconnects NVMe controller "
        "if hardware error detected. Refreshes file handles on dependent services. "
        "Use when storage-node reports ENOENT, ENOSPC, or kernel I/O errors."
    ),
    applicable_services=["storage-node", "order-service", "ml-inference"],
    trigger_condition=_rb_storage_remount_condition,
    effect=_rb_storage_remount_effect,
)

# ------------------------------------------------------------------

def _rb_ml_rollback_condition(gt: Dict) -> bool:
    root_svcs = gt.get("root_services", [gt.get("root_service", "")])
    if isinstance(root_svcs, str):
        root_svcs = [root_svcs]
    return "ml-inference" in root_svcs

def _rb_ml_rollback_effect(state: SimulatedState, gt: Dict) -> SimulatedState:
    state.apply_health("ml-inference", +0.90)
    state.stop_cascade_from("ml-inference")
    state.resolve_alerts_for("ml-inference", _alerts_by_service(gt))
    state.log("ml-inference rolled back to last stable model checkpoint")
    state.log("GPU memory cleared — CUDA context reset on all 4 workers")
    state.log("api-gateway ml route re-enabled")
    return state

RB_ML_MODEL_ROLLBACK = Runbook(
    id="rb_ml_model_rollback",
    name="ML Model Rollback to Last Stable Checkpoint",
    description=(
        "Rolls back ml-inference pods to the previous stable model version. "
        "Clears GPU memory, resets CUDA contexts, and restores the last known-good "
        "model checkpoint from storage. Re-enables traffic routing to ml-inference "
        "on api-gateway once health checks pass. "
        "Use when ml-inference shows OOM, CrashLoopBackOff, or serving latency spikes."
    ),
    applicable_services=["ml-inference", "api-gateway"],
    trigger_condition=_rb_ml_rollback_condition,
    effect=_rb_ml_rollback_effect,
)

# ------------------------------------------------------------------

def _rb_wrong_runbook_condition(gt: Dict) -> bool:
    # This runbook is NEVER appropriate — it represents a wrong action
    return False

def _rb_wrong_runbook_effect(state: SimulatedState, gt: Dict) -> SimulatedState:
    # Applying wrong runbook makes things slightly worse
    for svc in gt.get("involved_services", []):
        state.apply_health(svc, -0.05)
    state.log("WARNING: wrong runbook applied — unnecessary intervention may worsen state")
    return state

RB_WRONG_ACTION = Runbook(
    id="rb_wrong_action",
    name="Incorrect Remediation (Wrong Runbook Sentinel)",
    description=(
        "Sentinel runbook representing an incorrect agent action. "
        "Grader uses this to penalize wrong runbook selections. "
        "Never appropriate for any real incident."
    ),
    applicable_services=[],
    trigger_condition=_rb_wrong_runbook_condition,
    effect=_rb_wrong_runbook_effect,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class RunbookRegistry:
    """
    Lookup table for all runbooks.

    Usage:
        registry = RunbookRegistry()
        rb = registry.get("rb_db_failover")
        applicable = registry.get_applicable(ground_truth)
    """

    _ALL_RUNBOOKS: List[Runbook] = [
        RB_DB_FAILOVER,
        RB_SERVICE_RESTART,
        RB_CACHE_FLUSH_RESTART,
        RB_AUTH_TOKEN_INVALIDATE,
        RB_STORAGE_VOLUME_REMOUNT,
        RB_ML_MODEL_ROLLBACK,
        RB_WRONG_ACTION,
    ]

    def __init__(self) -> None:
        self._index: Dict[str, Runbook] = {rb.id: rb for rb in self._ALL_RUNBOOKS}

    def get(self, runbook_id: str) -> Runbook:
        if runbook_id not in self._index:
            raise KeyError(f"Unknown runbook: {runbook_id}. Available: {list(self._index)}")
        return self._index[runbook_id]

    def get_all(self) -> List[Runbook]:
        return list(self._ALL_RUNBOOKS)

    def get_applicable(self, ground_truth: Dict) -> List[Runbook]:
        """Return all runbooks whose trigger_condition passes for this incident."""
        return [rb for rb in self._ALL_RUNBOOKS if rb.is_applicable(ground_truth)]

    def list_ids(self) -> List[str]:
        return list(self._index.keys())
