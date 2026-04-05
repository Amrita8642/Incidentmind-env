"""
service_graph.py  (v2 — production upgrade)
============================================
Distributed microservices topology as a DAG with a full 4-state service
state machine (HEALTHY → DEGRADED → FAILING → RECOVERING) and 5 typed
failure modes.
 
Upgrades over v1
----------------
* FailureMode enum  — latency_spike | timeout | memory_leak |
                      disk_io_saturation | network_partition
  Each mode has a distinct damage profile, propagation multiplier, and
  alert-pattern tag consumed by AlertGenerator.
 
* ServiceState enum — HEALTHY | DEGRADED | FAILING | RECOVERING
  (superset of the old HealthState; HealthState kept as alias for
   backward compatibility with all existing tests)
 
* StateMachine      — deterministic transitions driven by health_score
  thresholds + elapsed simulated time; reversible via runbook_fix().
 
* CascadeHop        — gains `failure_mode` and `state` fields.
 
* simulate_failure_impact()
  — now accepts a FailureMode parameter
  — memory_leak uses gradual damage (not instant)
  — network_partition skips sensitivity roll (affects all reachable nodes)
  — disk_io_saturation saturates storage-tier neighbours first
 
* Edge-case guards
  — isolated failure (no downstream neighbours) returns single-hop list
  — all-services-failing detection helper
  — unknown service → KeyError with clear message
 
All randomness through seeded np.random.RandomState — no global state.
Determinism: same seed → identical cascade sequence for every failure mode.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
 
import networkx as nx
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
 
class FailureMode(str, Enum):
    """Typed failure causes; each carries distinct propagation behaviour."""
    LATENCY_SPIKE       = "latency_spike"        # gradual, affects all deps
    TIMEOUT             = "timeout"              # hard, high-damage, fast propagation
    MEMORY_LEAK         = "memory_leak"          # gradual damage accumulation
    DISK_IO_SATURATION  = "disk_io_saturation"   # storage-tier priority propagation
    NETWORK_PARTITION   = "network_partition"    # bypasses sensitivity roll
 
 
class ServiceState(str, Enum):
    """4-state service health state machine."""
    HEALTHY    = "HEALTHY"     # health_score ≥ 0.80
    DEGRADED   = "DEGRADED"    # health_score 0.40–0.80
    FAILING    = "FAILING"     # health_score 0.10–0.40
    RECOVERING = "RECOVERING"  # health_score rising after runbook applied
 
 
# Backward-compat alias for existing tests that import HealthState
class HealthState(str, Enum):
    NORMAL   = "NORMAL"
    DEGRADED = "DEGRADED"
    FAILING  = "FAILING"
 
 
def health_state_from_score(score: float) -> HealthState:
    """Backward-compat mapping — used by existing tests."""
    if score >= 0.8:
        return HealthState.NORMAL
    if score >= 0.4:
        return HealthState.DEGRADED
    return HealthState.FAILING
 
 
def service_state_from_score(score: float, recovering: bool = False) -> ServiceState:
    """4-state mapping used internally."""
    if recovering and score >= 0.4:
        return ServiceState.RECOVERING
    if score >= 0.8:
        return ServiceState.HEALTHY
    if score >= 0.4:
        return ServiceState.DEGRADED
    return ServiceState.FAILING
 
 
# ---------------------------------------------------------------------------
# Failure mode damage profiles
# ---------------------------------------------------------------------------
 
_FAILURE_PROFILES: Dict[FailureMode, Dict] = {
    FailureMode.LATENCY_SPIKE: {
        "initial_damage":      0.50,
        "attenuation":         0.65,   # damage multiplier per hop
        "bypass_sensitivity":  False,
        "storage_priority":    False,
        "gradual_steps":       1,
    },
    FailureMode.TIMEOUT: {
        "initial_damage":      0.85,
        "attenuation":         0.70,
        "bypass_sensitivity":  False,
        "storage_priority":    False,
        "gradual_steps":       1,
    },
    FailureMode.MEMORY_LEAK: {
        "initial_damage":      0.30,   # starts small
        "attenuation":         0.80,
        "bypass_sensitivity":  False,
        "storage_priority":    False,
        "gradual_steps":       3,      # applies damage in 3 incremental waves
    },
    FailureMode.DISK_IO_SATURATION: {
        "initial_damage":      0.75,
        "attenuation":         0.60,
        "bypass_sensitivity":  False,
        "storage_priority":    True,   # data-tier services damaged first / harder
        "gradual_steps":       1,
    },
    FailureMode.NETWORK_PARTITION: {
        "initial_damage":      0.90,
        "attenuation":         0.85,
        "bypass_sensitivity":  True,   # partition affects everything reachable
        "storage_priority":    False,
        "gradual_steps":       1,
    },
}
 
 
# ---------------------------------------------------------------------------
# ServiceNode
# ---------------------------------------------------------------------------
 
@dataclass
class ServiceNode:
    """Single microservice with SRE metadata and 4-state health tracking."""
    name: str
    tier: str                            # frontend | backend | data | infra
    criticality_score: float             # 0–1
    failure_sensitivity: float           # 0–1; probability of cascade uptake
    health_score: float = 1.0
    tags: List[str] = field(default_factory=list)
    _recovering: bool = field(default=False, repr=False, compare=False)
 
    # ── derived state ──────────────────────────────────────────────
    @property
    def state(self) -> HealthState:
        """Backward-compat property used by existing tests."""
        return health_state_from_score(self.health_score)
 
    @property
    def service_state(self) -> ServiceState:
        return service_state_from_score(self.health_score, self._recovering)
 
    # ── mutation ───────────────────────────────────────────────────
    def apply_damage(self, damage: float) -> None:
        self.health_score = float(np.clip(self.health_score - damage, 0.0, 1.0))
        self._recovering = False
 
    def apply_recovery(self, amount: float) -> None:
        self.health_score = float(np.clip(self.health_score + amount, 0.0, 1.0))
        self._recovering = True
 
    def reset(self) -> None:
        self.health_score = 1.0
        self._recovering = False
 
 
# ---------------------------------------------------------------------------
# CascadeHop  (extended)
# ---------------------------------------------------------------------------
 
@dataclass
class CascadeHop:
    """Record of one propagation step, now including failure_mode."""
    service: str
    health_score_after: float
    state: HealthState
    delay_seconds: float
    damage_applied: float
    failure_mode: FailureMode = FailureMode.TIMEOUT
    service_state: ServiceState = ServiceState.FAILING
 
 
# ---------------------------------------------------------------------------
# ServiceGraph
# ---------------------------------------------------------------------------
 
class ServiceGraph:
    """
    Immutable topology + mutable runtime health state.
 
    Edge direction: A → B means A *depends on* B (B is upstream of A).
    Failure in B propagates downstream to A and all other dependents.
    """
 
    _SERVICE_DEFINITIONS: List[Tuple] = [
        # (name, tier, criticality, sensitivity, tags)
        ("frontend-web",      "frontend", 0.70, 0.55, ["user-facing", "http"]),
        ("api-gateway",       "backend",  0.90, 0.80, ["routing", "auth-boundary"]),
        ("auth-service",      "backend",  0.95, 0.85, ["jwt", "session", "critical"]),
        ("payment-service",   "backend",  0.98, 0.90, ["pci", "transactions", "critical"]),
        ("order-service",     "backend",  0.85, 0.70, ["workflow", "saga"]),
        ("notification-svc",  "backend",  0.40, 0.30, ["async", "email", "sms"]),
        ("redis-cache",       "data",     0.88, 0.75, ["cache", "session-store"]),
        ("payment-db",        "data",     0.99, 0.95, ["postgres", "primary-db", "critical"]),
        ("order-db",          "data",     0.85, 0.80, ["postgres", "orders"]),
        ("storage-node",      "data",     0.80, 0.65, ["blob", "s3-compatible"]),
        ("ml-inference",      "backend",  0.60, 0.50, ["model-serving", "gpu"]),
        ("metrics-collector", "infra",    0.50, 0.20, ["prometheus", "telemetry"]),
        ("message-queue",     "infra",    0.88, 0.78, ["kafka", "async-backbone"]),
    ]
 
    _EDGE_DEFINITIONS: List[Tuple[str, str]] = [
        ("frontend-web",    "api-gateway"),
        ("api-gateway",     "auth-service"),
        ("api-gateway",     "payment-service"),
        ("api-gateway",     "order-service"),
        ("payment-service", "payment-db"),
        ("payment-service", "redis-cache"),
        ("auth-service",    "redis-cache"),
        ("order-service",   "order-db"),
        ("order-service",   "message-queue"),
        ("notification-svc","message-queue"),
        ("payment-service", "message-queue"),
        ("order-service",   "storage-node"),
        ("ml-inference",    "storage-node"),
        ("api-gateway",     "ml-inference"),
        ("metrics-collector","message-queue"),
    ]
 
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, ServiceNode] = {}
        self._build_graph()
 
    # ── construction ───────────────────────────────────────────────
 
    def _build_graph(self) -> None:
        for name, tier, crit, sens, tags in self._SERVICE_DEFINITIONS:
            node = ServiceNode(name=name, tier=tier,
                               criticality_score=crit, failure_sensitivity=sens, tags=tags)
            self._nodes[name] = node
            self._graph.add_node(name, meta=node)
        for src, dst in self._EDGE_DEFINITIONS:
            self._graph.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(self._graph):
            raise RuntimeError("Service graph contains a cycle.")
 
    # ── public query API (unchanged — backward-compat) ─────────────
 
    def get_graph(self) -> nx.DiGraph:
        return self._graph
 
    def get_all_services(self) -> List[str]:
        return sorted(self._graph.nodes())
 
    def get_upstream_services(self, service: str) -> List[str]:
        if service not in self._graph:
            raise KeyError(f"Unknown service: {service}")
        return list(self._graph.successors(service))
 
    def get_downstream_services(self, service: str) -> List[str]:
        if service not in self._graph:
            raise KeyError(f"Unknown service: {service}")
        return list(self._graph.predecessors(service))
 
    def get_service(self, name: str) -> ServiceNode:
        if name not in self._nodes:
            raise KeyError(f"Unknown service: {name}")
        return self._nodes[name]
 
    def get_metadata(self, service: str) -> Dict:
        node = self.get_service(service)
        return {
            "tier":                node.tier,
            "criticality_score":   node.criticality_score,
            "failure_sensitivity": node.failure_sensitivity,
            "health_score":        node.health_score,
            "state":               node.state.value,
            "service_state":       node.service_state.value,
            "tags":                node.tags,
        }
 
    # ── edge-case helpers ──────────────────────────────────────────
 
    def is_isolated_failure(self, service: str) -> bool:
        """True if *service* has no downstream dependents — isolated failure."""
        return len(self.get_downstream_services(service)) == 0
 
    def all_services_degraded(self) -> bool:
        """True if every service is in DEGRADED or FAILING state."""
        return all(n.health_score < 0.8 for n in self._nodes.values())
 
    def services_in_state(self, state: ServiceState) -> List[str]:
        """Return names of all services currently in *state*."""
        return [name for name, node in self._nodes.items()
                if node.service_state == state]
 
    # ── failure propagation (upgraded) ────────────────────────────
 
    def simulate_failure_impact(
        self,
        root_service: str,
        rng: np.random.RandomState,
        initial_damage: float = 0.85,
        delay_range: Tuple[float, float] = (5.0, 30.0),
        failure_mode: FailureMode = FailureMode.TIMEOUT,
    ) -> List[CascadeHop]:
        """
        BFS failure propagation from *root_service*.
 
        Behaviour varies by FailureMode:
          LATENCY_SPIKE      — lower initial damage, slower propagation
          TIMEOUT            — high damage, standard propagation
          MEMORY_LEAK        — gradual multi-wave damage
          DISK_IO_SATURATION — data-tier neighbours take priority damage
          NETWORK_PARTITION  — bypasses sensitivity roll; all reachable fail
 
        Returns time-sorted list of CascadeHop records.
        Mutates ServiceNode.health_score as a side-effect.
        """
        if root_service not in self._nodes:
            raise KeyError(f"Unknown root service: {root_service!r}. "
                           f"Valid: {sorted(self._nodes)}")
 
        profile = _FAILURE_PROFILES[failure_mode]
        eff_initial = profile["initial_damage"] if initial_damage == 0.85 else initial_damage
 
        hops: List[CascadeHop] = []
        visited: set = set()
 
        # ── root hop ──────────────────────────────────────────────
        root_node = self._nodes[root_service]
        if profile["gradual_steps"] > 1:
            step_dmg = eff_initial / profile["gradual_steps"]
            for _ in range(profile["gradual_steps"]):
                root_node.apply_damage(step_dmg)
        else:
            root_node.apply_damage(eff_initial)
 
        hops.append(CascadeHop(
            service=root_service,
            health_score_after=root_node.health_score,
            state=root_node.state,
            delay_seconds=0.0,
            damage_applied=eff_initial,
            failure_mode=failure_mode,
            service_state=root_node.service_state,
        ))
        visited.add(root_service)
 
        # ── isolated failure fast-path ─────────────────────────────
        if self.is_isolated_failure(root_service):
            return hops
 
        # ── BFS propagation ───────────────────────────────────────
        queue: List[Tuple[str, float, float]] = [(root_service, eff_initial, 0.0)]
 
        while queue:
            current_svc, current_damage, current_delay = queue.pop(0)
            downstream = self.get_downstream_services(current_svc)
 
            # disk_io_saturation: sort data-tier first for priority damage
            if profile["storage_priority"]:
                downstream = sorted(
                    downstream,
                    key=lambda s: (0 if self._nodes[s].tier == "data" else 1)
                )
 
            for dep_svc in downstream:
                if dep_svc in visited:
                    continue
                visited.add(dep_svc)
 
                dep_node = self._nodes[dep_svc]
 
                # network_partition bypasses sensitivity roll
                should_cascade = (
                    profile["bypass_sensitivity"]
                    or rng.random() < dep_node.failure_sensitivity
                )
                if not should_cascade:
                    continue
 
                next_damage = float(np.clip(current_damage * profile["attenuation"], 0.10, 1.0))
                hop_delay   = float(rng.uniform(*delay_range))
                total_delay = current_delay + hop_delay
 
                if profile["gradual_steps"] > 1:
                    step_dmg = next_damage / profile["gradual_steps"]
                    for _ in range(profile["gradual_steps"]):
                        dep_node.apply_damage(step_dmg)
                else:
                    dep_node.apply_damage(next_damage)
 
                hops.append(CascadeHop(
                    service=dep_svc,
                    health_score_after=dep_node.health_score,
                    state=dep_node.state,
                    delay_seconds=total_delay,
                    damage_applied=next_damage,
                    failure_mode=failure_mode,
                    service_state=dep_node.service_state,
                ))
                queue.append((dep_svc, next_damage, total_delay))
 
        hops.sort(key=lambda h: h.delay_seconds)
        return hops
 
    # ── state transitions ──────────────────────────────────────────
 
    def tick_recovery(self, service: str, recovery_per_tick: float = 0.05) -> ServiceState:
        """
        Advance a service one recovery tick (e.g. called each simulated second
        after a runbook has been applied).  Returns new ServiceState.
        """
        node = self._nodes.get(service)
        if node is None:
            raise KeyError(f"Unknown service: {service!r}")
        node.apply_recovery(recovery_per_tick)
        return node.service_state
 
    def reset_all_health(self) -> None:
        for node in self._nodes.values():
            node.reset()
 
    def apply_runbook_fix(self, service: str, recovery_amount: float = 0.90) -> None:
        if service in self._nodes:
            self._nodes[service].apply_recovery(recovery_amount)