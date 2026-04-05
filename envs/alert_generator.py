"""
alert_generator.py  (v2 — production upgrade)
===============================================
Generates a deterministic, temporally-ordered List[Alert] from an
IncidentScenario.  All v1 behaviour is preserved; this version adds:
 
New Alert fields
----------------
fingerprint_id   — stable hash of (service, alert_type, failure_mode).
                   Same root issue always gets the same fingerprint
                   across different seeds.
group_key        — grouping key for deduplication.  All alerts from
                   the same burst/root share one group_key.
occurrence_count — 1 for unique alerts; >1 for burst duplicates.
is_burst         — True if this alert is a burst duplicate.
is_flapping      — True if this alert represents a flap re-fire.
failure_mode     — FailureMode tag driving realistic message style.
 
New alert categories
--------------------
Burst alerts
  Rapid-fire duplicates from the same root service within seconds.
  They share group_key and fingerprint_id with their canonical parent.
  occurrence_count increments with each duplicate.
 
Flapping alerts
  Appear once, clear at T+Δ, then re-fire at T+2Δ.
  Stored as a single Alert with is_flapping=True; the AlertGenerator
  also inserts a synthetic "CLEAR" observation entry into the stream.
 
Smart red herrings (v2)
  Now share the same upstream dependency as the real root cause AND
  carry misleadingly similar timestamps (within ±10s of root cause).
  Their dependency_context shows the shared upstream — making them
  look *statistically correlated* without being causal.
 
Enhanced partial observability
  After INVESTIGATE the agent receives:
    - related_services   (topological neighbours)
    - dependency_context (upstream/downstream metadata)
    - failure_mode_hint  (NEW: broad category, not exact mode)
    - dedup_signal       (NEW: fingerprint + group + occurrence count)
  These together let an agent distinguish burst noise from new root cause.
 
Edge cases handled
------------------
  - Zero noise (task 1): no noise alerts generated, noise functions short-circuit
  - Identical timestamps: root cause + red herring at T=0 are both present;
    sort is stable (CRITICAL before HIGH for ties)
  - Missing cascade messages: graceful fallback to generic template
  - Alert storm (>100 alerts): generate() is O(N) and handles any N
  - All failure modes: message templates keyed on FailureMode
 
Determinism guarantee
---------------------
  Same seed → same fingerprint_id, group_key, occurrence_count, timestamps,
  burst offsets, and flap offsets for every task.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
 
import numpy as np
 
from .incident_generator import IncidentScenario
from .service_graph import FailureMode
 
 
# ---------------------------------------------------------------------------
# AlertSeverity
# ---------------------------------------------------------------------------
 
class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
 
 
# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------
 
def _make_fingerprint(service: str, alert_type: str,
                      failure_mode: FailureMode) -> str:
    """
    Stable, seed-independent fingerprint based on the alert's structural
    identity (service + type + failure_mode).  Same root issue → same
    fingerprint, regardless of when or how many times it fires.
 
    Uses a djb2-style hash for reproducibility without hashlib dependency.
    """
    raw = f"{service}|{alert_type}|{failure_mode.value}"
    h = 5381
    for ch in raw:
        h = ((h << 5) + h) + ord(ch)
        h &= 0xFFFFFFFF
    return f"fp_{h:08x}"
 
 
def _make_group_key(service: str, failure_mode: FailureMode) -> str:
    """
    Group key for deduplication — same service + failure mode = same group.
    Multiple instances of the same service firing in a burst share this key.
    """
    return f"{service.replace('-', '_')}_{failure_mode.value}"
 
 
# ---------------------------------------------------------------------------
# Alert  (extended)
# ---------------------------------------------------------------------------
 
@dataclass
class Alert:
    """
    SRE alert as seen in PagerDuty / OpsGenie / Datadog.
 
    Partial-observability contract
    --------------------------------
    Visible immediately (investigated=False):
      id, severity, source_service, alert_type, message,
      timestamp_offset, fingerprint_id, group_key, occurrence_count,
      is_burst, is_flapping
 
    Hidden until INVESTIGATE:
      related_services, dependency_context, failure_mode_hint, dedup_signal
    """
    id: str
    severity: AlertSeverity
    source_service: str
    alert_type: str
    message: str
    timestamp_offset: float
    is_noise: bool
    is_root_cause: bool
    is_red_herring: bool = False
    is_cascade: bool = False
    is_burst: bool = False
    is_flapping: bool = False
 
    # ── dedup / fingerprint fields (always visible) ────────────────
    fingerprint_id: str = ""
    group_key: str = ""
    occurrence_count: int = 1
 
    # ── partial observability (hidden until INVESTIGATE) ──────────
    related_services: List[str] = field(default_factory=list)
    dependency_context: Dict[str, str] = field(default_factory=dict)
    failure_mode_hint: str = ""     # broad hint, not exact mode
    dedup_signal: Dict[str, object] = field(default_factory=dict)
 
    # ── observation API ────────────────────────────────────────────
 
    def to_observation(self, investigated: bool = False) -> Dict:
        """
        Return agent-visible dict.
        Hidden fields are masked when investigated=False.
        """
        obs: Dict = {
            "id":               self.id,
            "severity":         self.severity.value,
            "source_service":   self.source_service,
            "alert_type":       self.alert_type,
            "message":          self.message,
            "timestamp_offset": self.timestamp_offset,
            "fingerprint_id":   self.fingerprint_id,
            "group_key":        self.group_key,
            "occurrence_count": self.occurrence_count,
            "is_burst":         self.is_burst,
            "is_flapping":      self.is_flapping,
            "is_noise":         None,       # ground truth hidden
            "is_root_cause":    None,
        }
        if investigated:
            obs["related_services"]   = self.related_services
            obs["dependency_context"] = self.dependency_context
            obs["failure_mode_hint"]  = self.failure_mode_hint
            obs["dedup_signal"]       = self.dedup_signal
        else:
            obs["related_services"]   = "[REDACTED — use INVESTIGATE]"
            obs["dependency_context"] = "[REDACTED — use INVESTIGATE]"
            obs["failure_mode_hint"]  = "[REDACTED — use INVESTIGATE]"
            obs["dedup_signal"]       = "[REDACTED — use INVESTIGATE]"
        return obs
 
 
# ---------------------------------------------------------------------------
# Message templates — keyed on (service, FailureMode) where possible
# ---------------------------------------------------------------------------
 
# Root cause messages: outer key = service, inner key = FailureMode value
_RC_MESSAGES: Dict[str, Dict[str, str]] = {
    "payment-db": {
        FailureMode.DISK_IO_SATURATION.value:
            "CRITICAL: payment-db disk I/O wait 98% — NVMe saturated, writes queuing (iowait avg 12s)",
        FailureMode.TIMEOUT.value:
            "CRITICAL: payment-db primary replica unreachable — connection pool exhausted (0/50)",
        FailureMode.MEMORY_LEAK.value:
            "CRITICAL: payment-db shared_buffers exhausted — OOM killer invoked on postgres pid",
        FailureMode.NETWORK_PARTITION.value:
            "CRITICAL: payment-db cluster network partition — replica unreachable from primary",
        FailureMode.LATENCY_SPIKE.value:
            "CRITICAL: payment-db replication lag 120s — failover threshold exceeded",
        "_default":
            "CRITICAL: payment-db unreachable — health check failing",
    },
    "redis-cache": {
        FailureMode.TIMEOUT.value:
            "CRITICAL: redis-cache connection refused port 6379 — process exited unexpectedly",
        FailureMode.MEMORY_LEAK.value:
            "CRITICAL: redis-cache OOM — maxmemory policy evicting session keys under load",
        FailureMode.NETWORK_PARTITION.value:
            "CRITICAL: redis-cache cluster partitioned — 3/6 shards unreachable",
        "_default":
            "CRITICAL: redis-cache cluster unhealthy — multiple shards down",
    },
    "storage-node": {
        FailureMode.DISK_IO_SATURATION.value:
            "CRITICAL: storage-node-01 NVMe controller error — disk I/O saturated, 0 write capacity",
        FailureMode.MEMORY_LEAK.value:
            "CRITICAL: storage-node-01 kernel OOM — page cache eviction causing I/O storm",
        FailureMode.TIMEOUT.value:
            "CRITICAL: storage-node volume mount failed — ENOENT on /data/blobs",
        "_default":
            "CRITICAL: storage-node kernel panic — node rebooting",
    },
    "ml-inference": {
        FailureMode.MEMORY_LEAK.value:
            "CRITICAL: ml-inference GPU memory leak — CUDA OOM on all 4 workers, pods restarting",
        FailureMode.TIMEOUT.value:
            "CRITICAL: ml-inference serving pods CrashLoopBackOff — 0/3 ready",
        FailureMode.LATENCY_SPIKE.value:
            "CRITICAL: ml-inference p99 latency >30s — health check failing, traffic dropped",
        "_default":
            "CRITICAL: ml-inference all workers unavailable",
    },
}
 
# Cascade messages — keyed on service
_CASCADE_MESSAGES: Dict[str, List[str]] = {
    "payment-service": [
        "HIGH: payment-service DB connection timeouts — 82% requests failing (circuit breaker OPEN)",
        "HIGH: payment-service circuit breaker OPEN — downstream payment-db unreachable",
        "MEDIUM: payment-service error budget burn rate 14x — SLO breach in 4 minutes",
    ],
    "auth-service": [
        "HIGH: auth-service session cache miss rate 100% — token validation latency 8s",
        "HIGH: auth-service JWT validation failing — redis dependency unavailable",
        "MEDIUM: auth-service 401 error rate elevated — users being logged out unexpectedly",
    ],
    "api-gateway": [
        "HIGH: api-gateway upstream error rate >60% — 503s propagating to all clients",
        "MEDIUM: api-gateway p99 latency 12s — upstream services degraded",
        "MEDIUM: api-gateway circuit breakers OPEN on 3/5 upstream routes",
    ],
    "frontend-web": [
        "MEDIUM: frontend-web 5xx error rate 45% — CDN edge seeing upstream failures",
        "MEDIUM: frontend-web user-facing timeout rate elevated — SLO breach imminent",
        "LOW: frontend-web Apdex 0.52 — user experience degraded",
    ],
    "order-service": [
        "HIGH: order-service saga orchestration stalled — storage dependency unavailable",
        "MEDIUM: order-service write throughput dropped 78% — storage I/O errors",
        "MEDIUM: order-service dead letter queue growing — failed sagas accumulating",
    ],
    "order-db": [
        "MEDIUM: order-db checkpoint lag growing — upstream write pressure reduced",
        "HIGH: order-db connection pool 90% utilized — downstream storage contention",
    ],
    "message-queue": [
        "MEDIUM: message-queue consumer lag 48k msgs — 3 consumer groups stalled",
        "HIGH: message-queue under-replicated partitions — ISR shrunk on 5 topics",
    ],
    "notification-svc": [
        "LOW: notification-svc dead letter queue 2.4k msgs — email delivery failing",
        "MEDIUM: notification-svc SMTP retry storm — upstream queue saturation",
    ],
}
 
# Noise messages: (service, alert_type, message)
_NOISE_MESSAGES: List[Tuple[str, str, str]] = [
    ("metrics-collector", "telemetry_gap",      "LOW: metrics-collector scrape interval missed — 2 data points lost"),
    ("notification-svc",  "queue_lag",           "LOW: notification-svc email queue lag 45s — within SLO tolerance"),
    ("frontend-web",      "cdn_miss",            "LOW: frontend-web CDN cache miss rate +3% — marginal traffic spike"),
    ("message-queue",     "consumer_lag",        "MEDIUM: message-queue consumer group lag 1200 msgs — within warning threshold"),
    ("order-service",     "retry_storm",         "LOW: order-service retry budget 60% consumed — monitoring"),
    ("metrics-collector", "high_cardinality",    "LOW: metrics-collector high-cardinality series — 12k series/min"),
    ("notification-svc",  "smtp_timeout",        "LOW: notification-svc SMTP connection soft timeout — retrying"),
    ("api-gateway",       "rate_limit_warning",  "MEDIUM: api-gateway rate limiter at 80% capacity — not yet dropping requests"),
    ("frontend-web",      "slow_asset",          "LOW: frontend-web static asset p95 220ms — elevated but not critical"),
    ("message-queue",     "partition_rebalance", "MEDIUM: message-queue partition rebalance triggered — brief consumer pause"),
    ("order-db",          "vacuum_running",      "LOW: order-db autovacuum running — minor I/O increase expected"),
    ("auth-service",      "token_cleanup",       "LOW: auth-service expired token cleanup job — normal maintenance"),
    ("storage-node",      "snapshot_lag",        "LOW: storage-node snapshot job delayed 5min — background I/O contention"),
    ("ml-inference",      "model_cache_warm",    "LOW: ml-inference model cache warming after deploy — latency elevated"),
    ("redis-cache",       "replication_pause",   "LOW: redis-cache AOF fsync slightly behind — within normal range"),
    ("payment-service",   "audit_log_delay",     "LOW: payment-service audit log write delayed 3s — non-blocking"),
    ("api-gateway",       "ssl_cert_warn",       "MEDIUM: api-gateway TLS cert expiring in 28 days — renew before deadline"),
    ("metrics-collector", "scrape_timeout",      "LOW: metrics-collector scrape timeout on 1/13 targets — transient"),
]
 
# Smart red herring messages — share upstream dependency context with real root cause
_RED_HERRING_MESSAGES: Dict[str, str] = {
    "metrics-collector": (
        "CRITICAL: metrics-collector reporting 100% packet loss to 6 services — "
        "possible network partition (telemetry pipeline issue; upstream connectivity intact)"
    ),
    "notification-svc": (
        "HIGH: notification-svc dead letter queue growing rapidly — "
        "suspected upstream storage failure (actual: queue backlog from deploy; storage healthy)"
    ),
    "frontend-web": (
        "HIGH: frontend-web Apdex score 0.42 — user experience severely degraded "
        "(actual: CDN misconfiguration; backend services responding normally)"
    ),
    "message-queue": (
        "HIGH: message-queue under-replicated partitions — ISR shrunk to 1 on 3 topics "
        "(actual: broker rolling restart; shares storage dependency with real incident)"
    ),
    "order-service": (
        "HIGH: order-service memory usage 94% — potential OOM risk "
        "(actual: expected after schema migration warm-up; same storage node as incident)"
    ),
}
 
 
# ---------------------------------------------------------------------------
# AlertGenerator
# ---------------------------------------------------------------------------
 
class AlertGenerator:
    """
    Converts an IncidentScenario into a fully-realized, time-sorted List[Alert].
 
    New in v2: also generates burst duplicates, flapping alerts, and emits
    synthetic CLEAR events for flapping alerts.  All behaviour is seeded and
    deterministic.
 
    Usage:
        ag     = AlertGenerator(seed=42)
        alerts = ag.generate(scenario)   # List[Alert], time-sorted
    """
 
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
 
    def generate(self, scenario: IncidentScenario) -> List[Alert]:
        """
        Deterministically generate all alerts for a scenario.
        Returns time-sorted list (ties broken: CRITICAL before HIGH/MEDIUM/LOW).
        """
        rng = np.random.RandomState(self._seed ^ (scenario.task_id * 0xFF01))
        alerts: List[Alert] = []
 
        alerts += self._build_root_cause_alerts(scenario, rng)
        alerts += self._build_cascade_alerts(scenario, rng)
        alerts += self._build_red_herring_alerts(scenario, rng)
        alerts += self._build_noise_alerts(scenario, rng)
        alerts += self._build_burst_alerts(scenario, rng)
        alerts += self._build_flapping_alerts(scenario, rng)
 
        _sev_rank = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH:     1,
            AlertSeverity.MEDIUM:   2,
            AlertSeverity.LOW:      3,
        }
        alerts.sort(key=lambda a: (a.timestamp_offset, _sev_rank[a.severity]))
        return alerts
 
    # ------------------------------------------------------------------
    # Root cause
    # ------------------------------------------------------------------
 
    def _build_root_cause_alerts(
        self, scenario: IncidentScenario, rng: np.random.RandomState
    ) -> List[Alert]:
        result = []
        root_svcs: List[str] = scenario.metadata.get(
            "root_services",
            [scenario.metadata.get("root_service", "")]
        )
        if isinstance(root_svcs, str):
            root_svcs = [root_svcs]
 
        for rc_id, svc in zip(scenario.root_cause_alert_ids, root_svcs):
            fmode = scenario.failure_mode
            svc_msgs = _RC_MESSAGES.get(svc, {})
            msg = svc_msgs.get(fmode.value, svc_msgs.get("_default",
                  f"CRITICAL: {svc} unreachable — health check failing"))
 
            fp  = _make_fingerprint(svc, "service_down", fmode)
            gk  = scenario.duplicate_group_map.get(rc_id, _make_group_key(svc, fmode))
 
            alert = Alert(
                id=rc_id,
                severity=AlertSeverity.CRITICAL,
                source_service=svc,
                alert_type="service_down",
                message=msg,
                timestamp_offset=0.0,
                is_noise=False,
                is_root_cause=True,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=1,
                related_services=self._get_related(svc, scenario),
                dependency_context=self._get_dep_context(svc, "root"),
                failure_mode_hint=self._failure_mode_hint(fmode),
                dedup_signal={"fingerprint": fp, "group": gk, "count": 1,
                              "is_canonical": True},
            )
            result.append(alert)
        return result
 
    # ------------------------------------------------------------------
    # Cascade
    # ------------------------------------------------------------------
 
    def _build_cascade_alerts(
        self, scenario: IncidentScenario, rng: np.random.RandomState
    ) -> List[Alert]:
        result = []
        for hop_idx, stage in enumerate(scenario.cascade_chain):
            svc  = stage.service
            msgs = _CASCADE_MESSAGES.get(svc, [
                f"HIGH: {svc} elevated error rate — upstream dependency failing",
            ])
            msg   = msgs[int(rng.randint(0, len(msgs)))]
            sev   = AlertSeverity.HIGH if hop_idx < 2 else AlertSeverity.MEDIUM
            fmode = stage.failure_mode
            fp    = _make_fingerprint(svc, "cascade_failure", fmode)
            gk    = _make_group_key(svc, fmode)
 
            alert = Alert(
                id=stage.alert_id,
                severity=sev,
                source_service=svc,
                alert_type="cascade_failure",
                message=msg,
                timestamp_offset=stage.delay_seconds,
                is_noise=False,
                is_root_cause=False,
                is_cascade=True,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=1,
                related_services=self._get_related(svc, scenario),
                dependency_context=self._get_dep_context(svc, "cascade"),
                failure_mode_hint=self._failure_mode_hint(fmode),
                dedup_signal={"fingerprint": fp, "group": gk, "count": 1,
                              "is_canonical": True},
            )
            result.append(alert)
        return result
 
    # ------------------------------------------------------------------
    # Red herrings — appear early, share upstream dependency
    # ------------------------------------------------------------------
 
    def _build_red_herring_alerts(
        self, scenario: IncidentScenario, rng: np.random.RandomState
    ) -> List[Alert]:
        result = []
        rh_services = {
            1: [],
            2: ["metrics-collector"],
            3: ["message-queue", "notification-svc", "frontend-web"],
        }
        services = rh_services.get(scenario.task_id, [])
 
        for rh_id, svc in zip(scenario.red_herring_alert_ids, services):
            # Smart correlation: timestamp within ±10s of T=0 (same time window as root)
            ts  = float(rng.uniform(0.0, 10.0))
            msg = _RED_HERRING_MESSAGES.get(svc, f"HIGH: {svc} anomaly — investigating")
            sev = AlertSeverity.HIGH if "HIGH" in msg else AlertSeverity.CRITICAL
            fp  = _make_fingerprint(svc, "anomaly", FailureMode.TIMEOUT)
            gk  = _make_group_key(svc, FailureMode.TIMEOUT)
 
            # dependency_context exposes shared upstream — smart red herring
            dep_ctx = self._get_dep_context(svc, "red_herring")
            dep_ctx["shared_upstream_with_incident"] = (
                "storage-node" if scenario.task_id == 3 else "redis-cache"
            )
            dep_ctx["correlation_note"] = (
                "Shares upstream dependency with root cause — "
                "investigate causal chain before acting"
            )
 
            alert = Alert(
                id=rh_id,
                severity=sev,
                source_service=svc,
                alert_type="anomaly",
                message=msg,
                timestamp_offset=ts,
                is_noise=False,
                is_root_cause=False,
                is_red_herring=True,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=1,
                related_services=[svc],
                dependency_context=dep_ctx,
                failure_mode_hint="unknown — investigate upstream",
                dedup_signal={"fingerprint": fp, "group": gk, "count": 1,
                              "is_canonical": True},
            )
            result.append(alert)
        return result
 
    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
 
    def _build_noise_alerts(
        self, scenario: IncidentScenario, rng: np.random.RandomState
    ) -> List[Alert]:
        result = []
        if not scenario.noise_alert_ids:
            return result
 
        if scenario.cascade_chain:
            max_delay = max(s.delay_seconds for s in scenario.cascade_chain)
        else:
            max_delay = 30.0
 
        n = len(scenario.noise_alert_ids)
        noise_timestamps = sorted(rng.uniform(0.0, max_delay + 20.0, size=n).tolist())
        pool_size = len(_NOISE_MESSAGES)
        indices   = list(rng.choice(pool_size, size=n, replace=True))
 
        for noise_id, ts, idx in zip(scenario.noise_alert_ids, noise_timestamps, indices):
            svc, atype, msg = _NOISE_MESSAGES[idx]
            sev_str = msg.split(":")[0].strip()
            sev = (AlertSeverity[sev_str]
                   if sev_str in AlertSeverity.__members__ else AlertSeverity.LOW)
            fp = _make_fingerprint(svc, atype, FailureMode.LATENCY_SPIKE)
            gk = _make_group_key(svc, FailureMode.LATENCY_SPIKE)
 
            alert = Alert(
                id=noise_id,
                severity=sev,
                source_service=svc,
                alert_type=atype,
                message=msg,
                timestamp_offset=round(ts, 2),
                is_noise=True,
                is_root_cause=False,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=1,
                related_services=[svc],
                dependency_context={"note": "routine — not incident related"},
                failure_mode_hint="routine_noise",
                dedup_signal={"fingerprint": fp, "group": gk, "count": 1,
                              "is_canonical": True},
            )
            result.append(alert)
        return result
 
    # ------------------------------------------------------------------
    # Burst alerts — rapid-fire duplicates of root cause
    # ------------------------------------------------------------------
 
    def _build_burst_alerts(
        self, scenario: IncidentScenario, rng: np.random.RandomState
    ) -> List[Alert]:
        """
        Generate burst duplicate alerts.  Each burst alert:
          - shares fingerprint_id and group_key with its canonical parent
          - has occurrence_count > 1 (incrementing per duplicate)
          - fires at a small positive offset from T=0 (same-service, rapid re-fire)
          - is_burst=True
        """
        result = []
        if not scenario.burst_alert_ids:
            return result
 
        # burst_offsets stored in scenario metadata; fall back to rng
        burst_offsets = scenario.metadata.get("burst_offsets", [])
        root_svcs: List[str] = scenario.metadata.get(
            "root_services",
            [scenario.metadata.get("root_service", "")]
        )
        if isinstance(root_svcs, str):
            root_svcs = [root_svcs]
        svc = root_svcs[0] if root_svcs else "unknown"
 
        fmode = scenario.failure_mode
        fp    = _make_fingerprint(svc, "service_down", fmode)
 
        for burst_idx, burst_id in enumerate(scenario.burst_alert_ids):
            gk  = scenario.duplicate_group_map.get(burst_id,
                   _make_group_key(svc, fmode))
            if burst_idx < len(burst_offsets):
                ts = burst_offsets[burst_idx]
            else:
                ts = round(float(rng.uniform(0.5, 3.0)), 2)
 
            occ = burst_idx + 2  # canonical is 1; first burst is 2, second is 3, etc.
 
            msg = (f"CRITICAL: {svc} — repeated failure signal "
                   f"(occurrence {occ}, fingerprint {fp[:10]}…)")
 
            alert = Alert(
                id=burst_id,
                severity=AlertSeverity.CRITICAL,
                source_service=svc,
                alert_type="service_down",
                message=msg,
                timestamp_offset=ts,
                is_noise=False,
                is_root_cause=False,    # burst is duplicate, not the root cause itself
                is_burst=True,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=occ,
                related_services=self._get_related(svc, scenario),
                dependency_context=self._get_dep_context(svc, "burst_duplicate"),
                failure_mode_hint=self._failure_mode_hint(fmode),
                dedup_signal={"fingerprint": fp, "group": gk, "count": occ,
                              "is_canonical": False, "canonical_id": scenario.root_cause_alert_ids[0]},
            )
            result.append(alert)
        return result
 
    # ------------------------------------------------------------------
    # Flapping alerts — appear once, clear, then re-fire
    # ------------------------------------------------------------------
 
    def _build_flapping_alerts(
        self, scenario: IncidentScenario, rng: np.random.RandomState
    ) -> List[Alert]:
        """
        Generate flapping alert.  For each flapping_alert_id we emit:
          1. The original fire  (is_flapping=True, occurrence_count=1)
          2. A synthetic CLEAR  (alert_type="clear", severity adjusted down)
          3. A re-fire          (is_flapping=True, occurrence_count=2)
 
        All three share the same fingerprint_id and group_key.
        They are interleaved at deterministic offsets within the cascade window.
        """
        result = []
        if not scenario.flapping_alert_ids:
            return result
 
        if scenario.cascade_chain:
            max_delay = max(s.delay_seconds for s in scenario.cascade_chain)
        else:
            max_delay = 60.0
 
        # Flapping services map (task_id → service list for flap alerts)
        flap_svc_map: Dict[int, List[str]] = {
            1: ["payment-service"],
            2: ["auth-service"],
            3: ["ml-inference"],
        }
        svc_list = flap_svc_map.get(scenario.task_id, ["api-gateway"])
 
        for flap_idx, flap_id in enumerate(scenario.flapping_alert_ids):
            svc   = svc_list[flap_idx % len(svc_list)]
            fmode = scenario.failure_mode
            fp    = _make_fingerprint(svc, "flapping", fmode)
            gk    = f"flap_{svc.replace('-', '_')}"
 
            # deterministic offsets
            t_fire  = round(float(rng.uniform(max_delay * 0.10, max_delay * 0.30)), 2)
            t_clear = round(t_fire + float(rng.uniform(5.0, 15.0)), 2)
            t_refire= round(t_clear + float(rng.uniform(10.0, 25.0)), 2)
 
            msgs = _CASCADE_MESSAGES.get(svc, [f"MEDIUM: {svc} intermittent failures"])
            msg_fire = msgs[0] + " [FLAPPING]"
            msg_clear = f"CLEAR: {svc} — alert resolved (may recur)"
            msg_refire = msgs[0] + " [FLAPPING — re-fired after clear]"
 
            dep_ctx = self._get_dep_context(svc, "flapping")
            dep_ctx["flap_note"] = (
                "This alert has cleared and re-fired — investigate root cause "
                "rather than treating as resolved"
            )
 
            # 1. Fire
            result.append(Alert(
                id=flap_id,
                severity=AlertSeverity.MEDIUM,
                source_service=svc,
                alert_type="flapping",
                message=msg_fire,
                timestamp_offset=t_fire,
                is_noise=False,
                is_root_cause=False,
                is_flapping=True,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=1,
                related_services=self._get_related(svc, scenario),
                dependency_context=dep_ctx,
                failure_mode_hint=self._failure_mode_hint(fmode),
                dedup_signal={"fingerprint": fp, "group": gk, "count": 1,
                              "flap_state": "firing"},
            ))
            # 2. Clear
            result.append(Alert(
                id=f"{flap_id}_clear",
                severity=AlertSeverity.LOW,
                source_service=svc,
                alert_type="clear",
                message=msg_clear,
                timestamp_offset=t_clear,
                is_noise=False,
                is_root_cause=False,
                is_flapping=True,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=1,
                related_services=[svc],
                dependency_context={"note": "clear event — alert may re-fire"},
                failure_mode_hint="clear",
                dedup_signal={"fingerprint": fp, "group": gk, "flap_state": "cleared"},
            ))
            # 3. Re-fire
            result.append(Alert(
                id=f"{flap_id}_refire",
                severity=AlertSeverity.HIGH,
                source_service=svc,
                alert_type="flapping",
                message=msg_refire,
                timestamp_offset=t_refire,
                is_noise=False,
                is_root_cause=False,
                is_flapping=True,
                fingerprint_id=fp,
                group_key=gk,
                occurrence_count=2,
                related_services=self._get_related(svc, scenario),
                dependency_context=dep_ctx,
                failure_mode_hint=self._failure_mode_hint(fmode),
                dedup_signal={"fingerprint": fp, "group": gk, "count": 2,
                              "flap_state": "refired"},
            ))
        return result
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    def _failure_mode_hint(self, fmode: FailureMode) -> str:
        """Broad hint revealed on INVESTIGATE — not the exact mode."""
        hints = {
            FailureMode.LATENCY_SPIKE:      "performance_degradation",
            FailureMode.TIMEOUT:            "connectivity_failure",
            FailureMode.MEMORY_LEAK:        "resource_exhaustion",
            FailureMode.DISK_IO_SATURATION: "storage_bottleneck",
            FailureMode.NETWORK_PARTITION:  "network_failure",
        }
        return hints.get(fmode, "unknown")
 
    def _get_related(self, service: str, scenario: IncidentScenario) -> List[str]:
        involved = set(scenario.involved_services)
        related  = {service}
        adjacency: Dict[str, List[str]] = {
            "payment-db":      ["payment-service"],
            "payment-service": ["payment-db", "redis-cache", "api-gateway"],
            "redis-cache":     ["auth-service", "payment-service"],
            "auth-service":    ["redis-cache", "api-gateway"],
            "api-gateway":     ["auth-service", "payment-service", "order-service", "frontend-web"],
            "frontend-web":    ["api-gateway"],
            "storage-node":    ["order-service", "ml-inference"],
            "order-service":   ["storage-node", "order-db", "message-queue"],
            "order-db":        ["order-service"],
            "ml-inference":    ["storage-node", "api-gateway"],
            "message-queue":   ["order-service", "notification-svc", "payment-service"],
            "metrics-collector": [],
            "notification-svc": ["message-queue"],
        }
        for adj in adjacency.get(service, []):
            if adj in involved:
                related.add(adj)
        return sorted(related)
 
    def _get_dep_context(self, service: str, role: str) -> Dict[str, str]:
        contexts: Dict[str, Dict] = {
            "payment-db": {
                "upstream_of": "payment-service",
                "tier": "data", "criticality": "0.99",
                "hint": "Check postgres replication and connection pool",
            },
            "redis-cache": {
                "upstream_of": "auth-service, payment-service",
                "tier": "data", "criticality": "0.88",
                "hint": "Shared session store — multiple services affected on failure",
            },
            "auth-service": {
                "upstream_of": "api-gateway", "depends_on": "redis-cache",
                "hint": "Token validation uses redis; cache failure propagates here",
            },
            "api-gateway": {
                "upstream_of": "frontend-web",
                "depends_on": "auth-service, payment-service, order-service",
                "hint": "Central routing — upstream errors aggregate here",
            },
            "storage-node": {
                "upstream_of": "order-service, ml-inference",
                "tier": "data",
                "hint": "Blob store shared by multiple workloads",
            },
            "ml-inference": {
                "upstream_of": "api-gateway", "depends_on": "storage-node",
                "hint": "Model artifacts on storage-node; storage failure → serving failure",
            },
        }
        ctx = dict(contexts.get(service, {"tier": "unknown", "hint": "Investigate dependencies"}))
        ctx["role"] = role
        return ctx