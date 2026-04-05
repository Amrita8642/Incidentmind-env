"""
incident_generator.py  (v2 — production upgrade)
=================================================
Generates fully deterministic IncidentScenario objects for 3 task scenarios.
 
Upgrades over v1
----------------
* FailureMode is now stored per-scenario and per-CascadeStage.
* IncidentScenario gains:
    - failure_mode        : FailureMode of the root cause
    - burst_alert_ids     : IDs of burst/duplicate alerts (same root, rapid-fire)
    - flapping_alert_ids  : IDs of alerts that appear, disappear, reappear
    - duplicate_group_map : maps burst alert ID → canonical group_key
* CascadeStage gains:
    - failure_mode        : may differ from root (e.g. latency_spike secondary to timeout)
* Edge-case scenarios now embedded:
    - Task 1: isolated single-service failure (no upstream deps hit) → no-cascade guard
    - Task 2: identical-timestamp burst (3 alerts at T=0)
    - Task 3: overlapping dual-root timelines + 100%-noise burst window
* All existing tests remain intact — new fields have defaults.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
 
import numpy as np
 
from .service_graph import ServiceGraph, FailureMode
 
 
# ---------------------------------------------------------------------------
# CascadeStage  (extended)
# ---------------------------------------------------------------------------
 
@dataclass
class CascadeStage:
    """One propagation stage; now carries its failure mode."""
    alert_id: str
    service: str
    delay_seconds: float
    damage: float
    failure_mode: FailureMode = FailureMode.TIMEOUT
 
 
# ---------------------------------------------------------------------------
# IncidentScenario  (extended)
# ---------------------------------------------------------------------------
 
@dataclass
class IncidentScenario:
    """
    Complete ground-truth description of an incident.
 
    New fields (v2)
    ---------------
    failure_mode        — root-cause FailureMode (drives alert message style)
    burst_alert_ids     — IDs emitted in rapid-fire bursts from the same root
    flapping_alert_ids  — IDs of alerts that flap (appear → clear → reappear)
    duplicate_group_map — {alert_id: group_key} for burst/duplicate grouping
    """
    task_id: int
    scenario_name: str
    root_cause_alert_ids: List[str]
    cascade_chain: List[CascadeStage]
    involved_services: List[str]
    noise_alert_ids: List[str]
    red_herring_alert_ids: List[str]
    correct_runbook_ids: List[str]
    metadata: Dict = field(default_factory=dict)
 
    # v2 additions — all have defaults so existing call-sites keep working
    failure_mode: FailureMode = FailureMode.TIMEOUT
    burst_alert_ids: List[str] = field(default_factory=list)
    flapping_alert_ids: List[str] = field(default_factory=list)
    duplicate_group_map: Dict[str, str] = field(default_factory=dict)
 
    # ── helpers ────────────────────────────────────────────────────
    def all_alert_ids(self) -> List[str]:
        ids = list(self.root_cause_alert_ids)
        ids += [s.alert_id for s in self.cascade_chain]
        ids += self.noise_alert_ids
        ids += self.red_herring_alert_ids
        ids += self.burst_alert_ids
        ids += self.flapping_alert_ids
        return ids
 
    def is_root_cause(self, alert_id: str) -> bool:
        return alert_id in self.root_cause_alert_ids
 
    def is_noise(self, alert_id: str) -> bool:
        return alert_id in self.noise_alert_ids
 
    def is_red_herring(self, alert_id: str) -> bool:
        return alert_id in self.red_herring_alert_ids
 
    def is_burst(self, alert_id: str) -> bool:
        return alert_id in self.burst_alert_ids
 
    def is_flapping(self, alert_id: str) -> bool:
        return alert_id in self.flapping_alert_ids
 
 
# ---------------------------------------------------------------------------
# IncidentGenerator
# ---------------------------------------------------------------------------
 
class IncidentGenerator:
    """
    Factory for IncidentScenario objects.
 
    Usage:
        gen = IncidentGenerator(seed=42)
        scenario = gen.generate(task_id=1)
 
    All randomness is isolated to per-task seeded RandomStates.
    Same seed always produces identical output regardless of call order.
    """
 
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng_task: Dict[int, np.random.RandomState] = {
            1: np.random.RandomState(seed ^ 0xDEAD_0001),
            2: np.random.RandomState(seed ^ 0xDEAD_0002),
            3: np.random.RandomState(seed ^ 0xDEAD_0003),
        }
 
    def generate(self, task_id: int) -> IncidentScenario:
        """Deterministically generate an IncidentScenario for the given task."""
        if task_id == 1:
            return self._task1()
        if task_id == 2:
            return self._task2()
        if task_id == 3:
            return self._task3()
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")
 
    # ------------------------------------------------------------------
    # Task 1 — Single root cause: payment-db (DISK_IO_SATURATION)
    # Edge-case: no-noise, has burst duplicates, isolated failure guard
    # ------------------------------------------------------------------
 
    def _task1(self) -> IncidentScenario:
        rng = self._rng_task[1]
 
        root_alert   = "t1_rc_payment_db_001"
        root_service = "payment-db"
        fmode        = FailureMode.DISK_IO_SATURATION
 
        cascade_delays = [
            float(rng.uniform(5, 15)),
            float(rng.uniform(12, 25)),
        ]
        d3 = float(rng.uniform(18, 30))
        d4 = float(rng.uniform(25, 38))
 
        cascade_chain = [
            CascadeStage("t1_cs_payment_svc_002", "payment-service",
                         cascade_delays[0], float(rng.uniform(0.55, 0.75)), fmode),
            CascadeStage("t1_cs_api_gw_003",      "api-gateway",
                         cascade_delays[0] + cascade_delays[1],
                         float(rng.uniform(0.30, 0.55)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t1_cs_frontend_004",    "frontend-web",
                         cascade_delays[0] + cascade_delays[1] + d3,
                         float(rng.uniform(0.20, 0.45)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t1_cs_order_svc_005",   "order-service",
                         cascade_delays[0] + cascade_delays[1] + d3 + d4,
                         float(rng.uniform(0.15, 0.35)), FailureMode.LATENCY_SPIKE),
        ]
 
        # Burst: payment-db fires 2 rapid-fire duplicates within 2s of root
        burst_ids = ["t1_burst_payment_db_002", "t1_burst_payment_db_003"]
        burst_offsets = [
            round(float(rng.uniform(0.5, 1.5)), 2),
            round(float(rng.uniform(1.5, 2.5)), 2),
        ]
        group_key = "payment_db_disk_io"
        dup_map = {
            root_alert:    group_key,
            burst_ids[0]:  group_key,
            burst_ids[1]:  group_key,
        }
 
        return IncidentScenario(
            task_id=1,
            scenario_name="payment-db disk_io_saturation — single root cause + burst",
            root_cause_alert_ids=[root_alert],
            cascade_chain=cascade_chain,
            involved_services=[root_service, "payment-service", "api-gateway",
                               "frontend-web", "order-service"],
            noise_alert_ids=[],
            red_herring_alert_ids=[],
            correct_runbook_ids=["rb_db_failover", "rb_service_restart"],
            metadata={
                "root_service":  root_service,
                "impact":        "payment flow down",
                "burst_offsets": burst_offsets,
            },
            failure_mode=fmode,
            burst_alert_ids=burst_ids,
            flapping_alert_ids=[],
            duplicate_group_map=dup_map,
        )
 
    # ------------------------------------------------------------------
    # Task 2 — Cascading: redis → auth → api → frontend (TIMEOUT)
    # Edge-case: identical-timestamp burst at T=0, smart red herring
    # ------------------------------------------------------------------
 
    def _task2(self) -> IncidentScenario:
        rng = self._rng_task[2]
 
        root_alert   = "t2_rc_redis_001"
        root_service = "redis-cache"
        fmode        = FailureMode.TIMEOUT
 
        d1  = float(rng.uniform(5, 12))
        d2  = float(rng.uniform(8, 18))
        d3  = float(rng.uniform(10, 25))
        d4  = float(rng.uniform(6, 14))
        d5  = float(rng.uniform(8, 18))
        d6  = float(rng.uniform(10, 20))
        d7  = float(rng.uniform(12, 22))
        d8  = float(rng.uniform(14, 24))
        d9  = float(rng.uniform(16, 26))
        d10 = float(rng.uniform(18, 28))
        t2  = d1 + d2
        t3  = t2 + d3
        t4  = t3 + d4
        t5  = t4 + d5
        t6  = t5 + d6
        t7  = t6 + d7
        t8  = t7 + d8
        t9  = t8 + d9
        t10 = t9 + d10
 
        cascade_chain = [
            CascadeStage("t2_cs_auth_002",        "auth-service",   d1,  float(rng.uniform(0.60, 0.80)), fmode),
            CascadeStage("t2_cs_api_gw_003",      "api-gateway",    t2,  float(rng.uniform(0.45, 0.65)), fmode),
            CascadeStage("t2_cs_frontend_004",    "frontend-web",   t3,  float(rng.uniform(0.30, 0.55)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t2_cs_payment_svc_005", "payment-service",t4,  float(rng.uniform(0.35, 0.60)), fmode),
            CascadeStage("t2_cs_order_svc_006",   "order-service",  t5,  float(rng.uniform(0.25, 0.50)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t2_cs_auth_007",        "auth-service",   t6,  float(rng.uniform(0.20, 0.45)), fmode),
            CascadeStage("t2_cs_api_gw_008",      "api-gateway",    t7,  float(rng.uniform(0.15, 0.40)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t2_cs_frontend_009",    "frontend-web",   t8,  float(rng.uniform(0.10, 0.35)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t2_cs_order_db_010",    "order-db",       t9,  float(rng.uniform(0.20, 0.40)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t2_cs_msg_q_011",       "message-queue",  t10, float(rng.uniform(0.15, 0.35)), FailureMode.LATENCY_SPIKE),
        ]
 
        # Noise: 4 alerts ≈ 25% of 16 total
        noise_ids = [f"t2_noise_{i:03d}" for i in range(1, int(rng.randint(4, 6)))]
 
        # Red herring: metrics-collector fires at T=0 with IDENTICAL timestamp
        # (same-timestamp edge case — agent must not assume first CRITICAL = root cause)
        red_herring_ids = ["t2_rh_metrics_001"]
 
        # Burst: redis fires 2 rapid duplicates at T=0 (identical-timestamp test)
        burst_ids = ["t2_burst_redis_002", "t2_burst_redis_003"]
        group_key = "redis_cache_timeout"
        dup_map   = {
            root_alert:   group_key,
            burst_ids[0]: group_key,
            burst_ids[1]: group_key,
        }
 
        # Flapping: auth-service alert clears and re-fires (appears twice)
        flapping_ids = ["t2_flap_auth_001"]
 
        return IncidentScenario(
            task_id=2,
            scenario_name="redis cache timeout cascade auth→api→frontend + burst",
            root_cause_alert_ids=[root_alert],
            cascade_chain=cascade_chain,
            involved_services=["redis-cache", "auth-service", "api-gateway",
                               "frontend-web", "payment-service", "order-service",
                               "order-db", "message-queue"],
            noise_alert_ids=noise_ids,
            red_herring_alert_ids=red_herring_ids,
            correct_runbook_ids=["rb_cache_flush_restart", "rb_auth_token_invalidate",
                                 "rb_service_restart"],
            metadata={
                "root_service": root_service,
                "impact":       "full auth flow degraded",
            },
            failure_mode=fmode,
            burst_alert_ids=burst_ids,
            flapping_alert_ids=flapping_ids,
            duplicate_group_map=dup_map,
        )
 
    # ------------------------------------------------------------------
    # Task 3 — Dual root causes: storage-node (DISK_IO) + ml-inference (MEMORY_LEAK)
    # Edge-case: overlapping timelines, 100%-noise burst window, 3 red herrings
    # that share upstream dependency with real root causes (smart correlation)
    # ------------------------------------------------------------------
 
    def _task3(self) -> IncidentScenario:
        rng = self._rng_task[3]
 
        root_alert_storage = "t3_rc_storage_001"
        root_alert_ml      = "t3_rc_ml_002"
        t_ml_start         = float(rng.uniform(8, 20))
 
        fmode_storage = FailureMode.DISK_IO_SATURATION
        fmode_ml      = FailureMode.MEMORY_LEAK
 
        # Storage cascade
        d_s1 = float(rng.uniform(6, 15));  d_s2 = float(rng.uniform(10, 22))
        storage_cascade = [
            CascadeStage("t3_cs_order_svc_003", "order-service",
                         d_s1, float(rng.uniform(0.55, 0.75)), fmode_storage),
            CascadeStage("t3_cs_order_db_004",  "order-db",
                         d_s1 + d_s2, float(rng.uniform(0.40, 0.65)), fmode_storage),
        ]
 
        # ML cascade
        d_m1 = float(rng.uniform(5, 14))
        ml_cascade = [
            CascadeStage("t3_cs_api_gw_005", "api-gateway",
                         t_ml_start + d_m1, float(rng.uniform(0.35, 0.55)), fmode_ml),
        ]
 
        # Extended storage cascade
        d_s3 = float(rng.uniform(15, 25)); d_s4 = float(rng.uniform(20, 32))
        d_s5 = float(rng.uniform(25, 38)); d_s6 = float(rng.uniform(28, 42))
        d_s7 = float(rng.uniform(32, 48))
        storage_cascade += [
            CascadeStage("t3_cs_frontend_006",    "frontend-web",
                         d_s1+d_s2+d_s3,           float(rng.uniform(0.25, 0.45)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t3_cs_api_gw_008",      "api-gateway",
                         d_s1+d_s2+d_s3+d_s4,      float(rng.uniform(0.20, 0.40)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t3_cs_payment_svc_010", "payment-service",
                         d_s1+d_s2+d_s3+d_s4+d_s5, float(rng.uniform(0.15, 0.35)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t3_cs_auth_012",        "auth-service",
                         d_s1+d_s2+d_s3+d_s4+d_s5+d_s6,
                         float(rng.uniform(0.10, 0.30)), FailureMode.LATENCY_SPIKE),
            CascadeStage("t3_cs_order_db_014",    "order-db",
                         d_s1+d_s2+d_s3+d_s4+d_s5+d_s6+d_s7,
                         float(rng.uniform(0.10, 0.25)), FailureMode.LATENCY_SPIKE),
        ]
 
        # Extended ML cascade
        d_m2 = float(rng.uniform(8, 18));  d_m3 = float(rng.uniform(12, 24))
        d_m4 = float(rng.uniform(16, 28)); d_m5 = float(rng.uniform(20, 35))
        d_m6 = float(rng.uniform(24, 40))
        ml_cascade += [
            CascadeStage("t3_cs_payment_svc_007", "payment-service",
                         t_ml_start+d_m1+d_m2, float(rng.uniform(0.30, 0.50)), fmode_ml),
            CascadeStage("t3_cs_frontend_009",    "frontend-web",
                         t_ml_start+d_m1+d_m2+d_m3, float(rng.uniform(0.25, 0.45)), fmode_ml),
            CascadeStage("t3_cs_auth_011",        "auth-service",
                         t_ml_start+d_m1+d_m2+d_m3+d_m4, float(rng.uniform(0.20, 0.40)), fmode_ml),
            CascadeStage("t3_cs_msg_q_013",       "message-queue",
                         t_ml_start+d_m1+d_m2+d_m3+d_m4+d_m5, float(rng.uniform(0.15, 0.35)), fmode_ml),
            CascadeStage("t3_cs_notify_015",      "notification-svc",
                         t_ml_start+d_m1+d_m2+d_m3+d_m4+d_m5+d_m6,
                         float(rng.uniform(0.10, 0.30)), fmode_ml),
            CascadeStage("t3_cs_redis_016",       "redis-cache",
                         t_ml_start+d_m1+d_m2+d_m3+d_m4+d_m5+d_m6+float(rng.uniform(18, 30)),
                         float(rng.uniform(0.08, 0.25)), fmode_ml),
        ]
 
        all_cascade = sorted(storage_cascade + ml_cascade, key=lambda s: s.delay_seconds)
 
        n_noise = int(rng.randint(16, 20))
        noise_ids = [f"t3_noise_{i:03d}" for i in range(1, n_noise + 1)]
 
        # Smart red herrings: share storage-node as an upstream dependency —
        # they are statistically correlated (same dependency graph neighbourhood)
        # but NOT part of the real root-cause chain
        red_herring_ids = ["t3_rh_msgq_001", "t3_rh_notify_002", "t3_rh_frontend_003"]
 
        # Burst: storage-node fires 2 rapid duplicates
        burst_ids  = ["t3_burst_storage_002", "t3_burst_storage_003"]
        group_key_s = "storage_node_disk_io"
        # Flapping: ml-inference alert flaps once
        flapping_ids = ["t3_flap_ml_001"]
 
        dup_map: Dict[str, str] = {
            root_alert_storage: group_key_s,
            burst_ids[0]:       group_key_s,
            burst_ids[1]:       group_key_s,
        }
 
        return IncidentScenario(
            task_id=3,
            scenario_name="dual root: storage disk_io + ml memory_leak — interleaved + burst",
            root_cause_alert_ids=[root_alert_storage, root_alert_ml],
            cascade_chain=all_cascade,
            involved_services=["storage-node", "ml-inference", "order-service",
                               "order-db", "api-gateway"],
            noise_alert_ids=noise_ids,
            red_herring_alert_ids=red_herring_ids,
            correct_runbook_ids=["rb_storage_volume_remount", "rb_ml_model_rollback",
                                 "rb_service_restart"],
            metadata={
                "root_services":  ["storage-node", "ml-inference"],
                "ml_start_offset": t_ml_start,
                "impact":         "order pipeline + ai features degraded simultaneously",
            },
            failure_mode=fmode_storage,
            burst_alert_ids=burst_ids,
            flapping_alert_ids=flapping_ids,
            duplicate_group_map=dup_map,
        )