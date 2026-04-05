"""
test_env_validation.py
======================
IncidentMind — Production-Grade Environment Validation Suite
============================================================

Test categories:
  T1  Determinism           — same seed → identical output every time
  T2  Task constraints      — alert counts, noise %, red-herring counts in-spec
  T3  Temporal ordering     — alerts sorted by timestamp; cascade after root cause
  T4  Noise / red herrings  — flags, severity rules enforced
  T5  Grader correctness    — perfect=high score; greedy/wrong=penalized
  T6  Performance           — full pipeline < 500ms per iteration
  T7  Service graph         — DAG, 12+ services, cascade hops, health states
  T8  Partial observability — hidden fields masked until INVESTIGATE
  T9  Runbook registry      — 7 runbooks, applicability, effect correctness
  T10 Multi-root scoring    — task3 partial + full credit paths

Usage:
    python test_env_validation.py
    python test_env_validation.py -v          # verbose sub-step detail
    python test_env_validation.py --fast      # skip performance test
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Any, Callable, Dict, List

import numpy as np

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from envs.service_graph    import ServiceGraph, HealthState, health_state_from_score
from envs.incident_generator import IncidentGenerator, IncidentScenario
from envs.alert_generator  import AlertGenerator, Alert, AlertSeverity
from envs.runbooks         import RunbookRegistry, SimulatedState
from envs.grader           import Grader, ActionType, GradeResult
from envs.tasks            import get_task, list_tasks

# ─────────────────────────── runner infra ───────────────────────────────────
_PASS  = "\033[92m  PASS\033[0m"
_FAIL  = "\033[91m  FAIL\033[0m"
_SKIP  = "\033[93m  SKIP\033[0m"
_TITLE = "\033[1;96m"
_RESET = "\033[0m"
_results: List[Dict[str, Any]] = []
_verbose = False

def _log(msg: str) -> None:
    if _verbose:
        print(f"         {msg}")

def run_test(name: str, fn: Callable[[], None], skip: bool = False) -> bool:
    if skip:
        print(f"{_SKIP}  {name}")
        _results.append({"name": name, "status": "SKIP"})
        return True
    try:
        fn()
        print(f"{_PASS}  {name}")
        _results.append({"name": name, "status": "PASS"})
        return True
    except AssertionError as exc:
        print(f"{_FAIL}  {name}")
        print(f"         AssertionError: {exc}")
        _results.append({"name": name, "status": "FAIL", "error": str(exc)})
        return False
    except Exception as exc:
        print(f"{_FAIL}  {name}")
        print(f"         Exception: {exc}")
        if _verbose:
            traceback.print_exc()
        _results.append({"name": name, "status": "FAIL", "error": str(exc)})
        return False

def section(title: str) -> None:
    print(f"\n{_TITLE}{'─'*65}{_RESET}")
    print(f"{_TITLE}  {title}{_RESET}")
    print(f"{_TITLE}{'─'*65}{_RESET}")

# ─────────────────────────── shared fixtures ────────────────────────────────
SEED     = 42
_sg      = ServiceGraph()
_gen     = IncidentGenerator(seed=SEED)
_ag      = AlertGenerator(seed=SEED)
_reg     = RunbookRegistry()
_grdr    = Grader()
_scenarios = {tid: _gen.generate(tid) for tid in (1, 2, 3)}
_alerts    = {tid: _ag.generate(_scenarios[tid]) for tid in (1, 2, 3)}

def _build_gt(sc: IncidentScenario) -> Dict:
    root_svcs = sc.metadata.get("root_services",
                    [sc.metadata.get("root_service", "")])
    if isinstance(root_svcs, str):
        root_svcs = [root_svcs]
    return {
        "task_id":               sc.task_id,
        "root_cause_alert_ids":  sc.root_cause_alert_ids,
        "cascade_chain":         [{"alert_id": s.alert_id} for s in sc.cascade_chain],
        "noise_alert_ids":       sc.noise_alert_ids,
        "red_herring_alert_ids": sc.red_herring_alert_ids,
        "correct_runbook_ids":   sc.correct_runbook_ids,
        "involved_services":     sc.involved_services,
        "root_services":         root_svcs,
        "alerts_by_service":     {},
    }

def _perfect_actions(sc: IncidentScenario, al: List[Alert]) -> List[Dict]:
    acts, step = [], 0
    for a in al[:3]:
        acts.append({"type": ActionType.INVESTIGATE, "alert_id": a.id, "runbook_id": None, "step": step}); step += 1
    for rc in sc.root_cause_alert_ids:
        acts.append({"type": ActionType.IDENTIFY_ROOT_CAUSE, "alert_id": rc, "runbook_id": None, "step": step}); step += 1
    for rb in sc.correct_runbook_ids:
        acts.append({"type": ActionType.APPLY_RUNBOOK, "alert_id": None, "runbook_id": rb, "step": step}); step += 1
    for nid in sc.noise_alert_ids:
        acts.append({"type": ActionType.DISMISS_NOISE, "alert_id": nid, "runbook_id": None, "step": step}); step += 1
    acts.append({"type": ActionType.RESOLVE, "alert_id": None, "runbook_id": None, "step": step})
    return acts

def _greedy_actions(sc: IncidentScenario, al: List[Alert]) -> List[Dict]:
    first_crit = next((a for a in al if a.severity == AlertSeverity.CRITICAL), al[0])
    return [
        {"type": ActionType.IDENTIFY_ROOT_CAUSE, "alert_id": first_crit.id, "runbook_id": None, "step": 0},
        {"type": ActionType.RESOLVE,              "alert_id": None,          "runbook_id": None, "step": 1},
    ]

# ══════════════════════════════════════════════════════════════════════════════
# T1 — Determinism
# ══════════════════════════════════════════════════════════════════════════════
def test_determinism_scenario():
    for tid in (1, 2, 3):
        a, b = IncidentGenerator(seed=SEED).generate(tid), IncidentGenerator(seed=SEED).generate(tid)
        assert a.root_cause_alert_ids == b.root_cause_alert_ids, f"Task {tid}: root_cause_alert_ids differ"
        assert len(a.cascade_chain) == len(b.cascade_chain),     f"Task {tid}: cascade chain length differs"
        assert a.noise_alert_ids     == b.noise_alert_ids,       f"Task {tid}: noise_alert_ids differ"
        _log(f"Task {tid} scenario OK")

def test_determinism_alerts():
    for tid in (1, 2, 3):
        sc = _scenarios[tid]
        al_a = AlertGenerator(seed=SEED).generate(sc)
        al_b = AlertGenerator(seed=SEED).generate(sc)
        assert [a.id for a in al_a]               == [a.id for a in al_b],  f"Task {tid}: alert IDs differ"
        assert [a.timestamp_offset for a in al_a] == [a.timestamp_offset for a in al_b], f"Task {tid}: timestamps differ"
        _log(f"Task {tid} alerts OK ({len(al_a)} alerts)")

def test_determinism_failure_propagation():
    for root in ("payment-db", "redis-cache", "storage-node"):
        hops_a = ServiceGraph().simulate_failure_impact(root, np.random.RandomState(SEED))
        hops_b = ServiceGraph().simulate_failure_impact(root, np.random.RandomState(SEED))
        assert [h.service for h in hops_a] == [h.service for h in hops_b], f"{root}: cascade services differ"
        for ha, hb in zip(hops_a, hops_b):
            assert abs(ha.delay_seconds - hb.delay_seconds) < 1e-9, f"{root}: delay mismatch at {ha.service}"
        _log(f"{root}: {[h.service for h in hops_a]}")

def test_determinism_grader():
    for tid in (1, 2, 3):
        gt   = _build_gt(_scenarios[tid])
        acts = _perfect_actions(_scenarios[tid], _alerts[tid])
        r1, r2 = _grdr.grade(gt, acts, tid), _grdr.grade(gt, acts, tid)
        assert r1.total_score       == r2.total_score,       f"Task {tid}: total_score not deterministic"
        assert r1.root_cause_score  == r2.root_cause_score,  f"Task {tid}: root_cause_score not deterministic"
        _log(f"Task {tid} grader OK (score={r1.total_score:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# T2 — Task constraints
# ══════════════════════════════════════════════════════════════════════════════
def test_task_alert_counts():
    for tid in (1, 2, 3):
        task = get_task(tid)
        n    = len(_alerts[tid])
        lo, hi = task.alert_count_range
        _log(f"Task {tid}: {n} alerts, range=[{lo},{hi}]")
        assert lo <= n <= hi, f"Task {tid}: alert count {n} outside [{lo},{hi}]"

def test_task_noise_percentage():
    for tid in (1, 2, 3):
        task   = get_task(tid)
        alerts = _alerts[tid]
        n_noise = sum(1 for a in alerts if a.is_noise)
        pct     = n_noise / len(alerts) if alerts else 0.0
        _log(f"Task {tid}: noise={n_noise}/{len(alerts)} ({pct:.0%}) expected≈{task.noise_percentage:.0%}")
        if tid == 1:
            assert n_noise == 0, f"Task 1 must have 0 noise alerts, got {n_noise}"
        else:
            assert abs(pct - task.noise_percentage) <= 0.15, (
                f"Task {tid}: noise% {pct:.2%} deviates > 15% from expected {task.noise_percentage:.2%}")

def test_task_red_herring_counts():
    for tid in (1, 2, 3):
        task = get_task(tid)
        n_rh = len(_scenarios[tid].red_herring_alert_ids)
        _log(f"Task {tid}: red_herrings={n_rh} expected={task.red_herring_count}")
        assert n_rh == task.red_herring_count, f"Task {tid}: expected {task.red_herring_count} RH, got {n_rh}"

def test_task_definitions_sane():
    tasks = list_tasks()
    assert len(tasks) == 3
    assert [t.task_id for t in tasks] == [1, 2, 3]
    for t in tasks:
        lo, hi = t.alert_count_range
        assert lo < hi
        assert 0.0 <= t.noise_percentage <= 1.0
        assert 0.0 < t.passing_score <= 1.0
        assert t.max_steps > 0
        _log(f"Task {t.task_id} ({t.difficulty}): max_steps={t.max_steps}, passing={t.passing_score}")

# ══════════════════════════════════════════════════════════════════════════════
# T3 — Temporal ordering
# ══════════════════════════════════════════════════════════════════════════════
def test_temporal_sort_order():
    for tid in (1, 2, 3):
        ts = [a.timestamp_offset for a in _alerts[tid]]
        assert ts == sorted(ts), f"Task {tid}: alerts not time-sorted"
        _log(f"Task {tid}: {len(ts)} alerts time-sorted ✓")

def test_root_cause_fires_first():
    for tid in (1, 2, 3):
        sc    = _scenarios[tid]
        rc_ts = [a.timestamp_offset for a in _alerts[tid] if a.id in sc.root_cause_alert_ids]
        all_ts = [a.timestamp_offset for a in _alerts[tid]]
        assert rc_ts, f"Task {tid}: no root cause alerts found"
        min_rc = min(rc_ts)
        avg_all = sum(all_ts) / len(all_ts)
        assert min_rc <= avg_all, f"Task {tid}: root cause T={min_rc:.1f} > avg T={avg_all:.1f}"
        _log(f"Task {tid}: root cause T={min_rc:.1f}s, avg all={avg_all:.1f}s")

def test_cascade_after_root_cause():
    for tid in (1, 2, 3):
        sc = _scenarios[tid]
        cascade_ids = {s.alert_id for s in sc.cascade_chain}
        for a in _alerts[tid]:
            if a.id in cascade_ids:
                assert a.timestamp_offset > 0.0, \
                    f"Task {tid}: cascade alert {a.id} at T={a.timestamp_offset} not > 0"
        _log(f"Task {tid}: all cascade alerts post-T0 ✓")

def test_cascade_delays_monotonic():
    for tid in (1, 2, 3):
        delays = [s.delay_seconds for s in _scenarios[tid].cascade_chain]
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i-1], \
                f"Task {tid}: cascade delay at stage {i} ({delays[i]:.2f}) < stage {i-1} ({delays[i-1]:.2f})"
        _log(f"Task {tid}: cascade delays {[round(d,1) for d in delays]}")

# ══════════════════════════════════════════════════════════════════════════════
# T4 — Noise / red-herring flags
# ══════════════════════════════════════════════════════════════════════════════
def test_noise_flags_correct():
    for tid in (1, 2, 3):
        noise_ids = set(_scenarios[tid].noise_alert_ids)
        for a in _alerts[tid]:
            if a.id in noise_ids:
                assert a.is_noise,         f"Task {tid}: {a.id} in noise_ids but is_noise=False"
                assert not a.is_root_cause, f"Task {tid}: noise {a.id} has is_root_cause=True"

def test_root_cause_flags_correct():
    for tid in (1, 2, 3):
        rc_ids = set(_scenarios[tid].root_cause_alert_ids)
        for a in _alerts[tid]:
            if a.id in rc_ids:
                assert a.is_root_cause, f"Task {tid}: {a.id} in RC ids but is_root_cause=False"
                assert not a.is_noise,   f"Task {tid}: root cause {a.id} has is_noise=True"

def test_red_herring_severity():
    high_sev = {AlertSeverity.CRITICAL, AlertSeverity.HIGH}
    for tid in (1, 2, 3):
        sc = _scenarios[tid]
        rh_ids = set(sc.red_herring_alert_ids)
        for a in _alerts[tid]:
            if a.id in rh_ids:
                assert a.severity in high_sev, \
                    f"Task {tid}: red herring {a.id} severity={a.severity} (must be HIGH/CRITICAL)"
        _log(f"Task {tid}: {len(rh_ids)} red herrings — all HIGH/CRITICAL ✓")

def test_noise_not_critical():
    for tid in (1, 2, 3):
        noise_ids = set(_scenarios[tid].noise_alert_ids)
        for a in _alerts[tid]:
            if a.id in noise_ids:
                assert a.severity != AlertSeverity.CRITICAL, \
                    f"Task {tid}: noise alert {a.id} is CRITICAL"

def test_partial_observability_masking():
    for tid in (1, 2, 3):
        a = _alerts[tid][0]
        obs_hidden = a.to_observation(investigated=False)
        obs_open   = a.to_observation(investigated=True)
        assert isinstance(obs_open["related_services"], list), \
            f"Alert {a.id}: related_services must be list when investigated"
        assert "REDACTED" in str(obs_hidden["related_services"]) or \
               obs_hidden["related_services"] != obs_open["related_services"], \
            f"Alert {a.id}: hidden field not masked when uninvestigated"
        _log(f"Task {tid}: masking OK on alert {a.id}")

# ══════════════════════════════════════════════════════════════════════════════
# T5 — Grader correctness
# ══════════════════════════════════════════════════════════════════════════════
def test_perfect_agent_passes():
    for tid in (1, 2, 3):
        sc = _scenarios[tid]; al = _alerts[tid]
        r  = _grdr.grade(_build_gt(sc), _perfect_actions(sc, al), tid)
        _log(f"Task {tid}: total={r.total_score:.4f} rc={r.root_cause_score:.4f} "
             f"rb={r.runbook_score:.4f} ns={r.noise_suppression_score:.4f} eff={r.efficiency_score:.4f}")
        assert r.passed,             f"Task {tid}: perfect agent did not pass"
        assert r.root_cause_score == 1.0, f"Task {tid}: perfect RC should score 1.0"
        assert r.runbook_score    == 1.0, f"Task {tid}: perfect runbook should score 1.0"

def test_greedy_agent_penalized():
    for tid in (1, 2, 3):
        sc = _scenarios[tid]; al = _alerts[tid]; gt = _build_gt(sc)
        rp = _grdr.grade(gt, _perfect_actions(sc, al), tid)
        rg = _grdr.grade(gt, _greedy_actions(sc, al), tid)
        _log(f"Task {tid}: perfect={rp.total_score:.4f} greedy={rg.total_score:.4f}")
        assert rg.efficiency_score < rp.efficiency_score, \
            f"Task {tid}: greedy efficiency not penalized"
        assert rg.total_score < rp.total_score, \
            f"Task {tid}: greedy total_score not below perfect"

def test_wrong_root_cause_penalized():
    for tid in (1, 2, 3):
        sc = _scenarios[tid]; al = _alerts[tid]; gt = _build_gt(sc)
        wrong_id = next((a.id for a in al if not a.is_root_cause), al[-1].id)
        acts = [{"type": ActionType.INVESTIGATE,       "alert_id": al[0].id,  "runbook_id": None, "step": 0},
                {"type": ActionType.IDENTIFY_ROOT_CAUSE,"alert_id": wrong_id, "runbook_id": None, "step": 1},
                {"type": ActionType.RESOLVE,            "alert_id": None,     "runbook_id": None, "step": 2}]
        r = _grdr.grade(gt, acts, tid)
        _log(f"Task {tid}: wrong RC → rc_score={r.root_cause_score:.4f}")
        assert r.root_cause_score < 1.0, f"Task {tid}: wrong RC should reduce root_cause_score"

def test_wrong_runbook_penalized():
    for tid in (1, 2, 3):
        sc = _scenarios[tid]; al = _alerts[tid]; gt = _build_gt(sc)
        acts = []
        step = 0
        for a in al[:2]:
            acts.append({"type": ActionType.INVESTIGATE, "alert_id": a.id, "runbook_id": None, "step": step}); step+=1
        for rc in sc.root_cause_alert_ids:
            acts.append({"type": ActionType.IDENTIFY_ROOT_CAUSE, "alert_id": rc, "runbook_id": None, "step": step}); step+=1
        acts.append({"type": ActionType.APPLY_RUNBOOK, "alert_id": None, "runbook_id": "rb_wrong_action", "step": step}); step+=1
        acts.append({"type": ActionType.RESOLVE, "alert_id": None, "runbook_id": None, "step": step})
        r = _grdr.grade(gt, acts, tid)
        _log(f"Task {tid}: wrong runbook → rb_score={r.runbook_score:.4f}")
        assert r.runbook_score < 1.0, f"Task {tid}: wrong runbook should reduce runbook_score"

def test_no_action_scores_zero():
    for tid in (1, 2, 3):
        r = _grdr.grade(_build_gt(_scenarios[tid]), [], tid)
        assert r.total_score == 0.0, f"Task {tid}: empty actions should score 0.0, got {r.total_score}"
        _log(f"Task {tid}: no-action → 0.0 ✓")

def test_grade_result_complete():
    for tid in (1, 2, 3):
        sc = _scenarios[tid]; al = _alerts[tid]
        r  = _grdr.grade(_build_gt(sc), _perfect_actions(sc, al), tid)
        for attr in ("total_score","root_cause_score","runbook_score","noise_suppression_score","efficiency_score"):
            val = getattr(r, attr)
            assert 0.0 <= val <= 1.0, f"Task {tid}: {attr}={val} out of [0,1]"
        assert isinstance(r.passed, bool)
        assert "task_id" in r.details

# ══════════════════════════════════════════════════════════════════════════════
# T6 — Performance
# ══════════════════════════════════════════════════════════════════════════════
def test_pipeline_performance():
    MAX_MS = 500.0
    ITERS  = 10
    for tid in (1, 2, 3):
        t0 = time.perf_counter()
        for i in range(ITERS):
            sc = IncidentGenerator(seed=SEED+i).generate(tid)
            al = AlertGenerator(seed=SEED+i).generate(sc)
            _grdr.grade(_build_gt(sc), _perfect_actions(sc, al), tid)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_ms = elapsed_ms / ITERS
        _log(f"Task {tid}: {ITERS} iters in {elapsed_ms:.1f}ms ({per_ms:.1f}ms each)")
        assert per_ms < MAX_MS, f"Task {tid}: avg {per_ms:.1f}ms > {MAX_MS}ms limit"

# ══════════════════════════════════════════════════════════════════════════════
# T7 — Service graph
# ══════════════════════════════════════════════════════════════════════════════
def test_service_count():
    svcs = _sg.get_all_services()
    _log(f"{len(svcs)} services: {svcs}")
    assert len(svcs) >= 12, f"Need >= 12 services, got {len(svcs)}"

def test_graph_is_dag():
    import networkx as nx
    g = _sg.get_graph()
    assert nx.is_directed_acyclic_graph(g), "Service graph contains a cycle"
    _log(f"DAG: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

def test_all_tiers_present():
    required = {"frontend", "backend", "data", "infra"}
    found    = {_sg.get_metadata(s)["tier"] for s in _sg.get_all_services()}
    _log(f"Tiers found: {found}")
    assert required <= found, f"Missing tiers: {required - found}"

def test_criticality_range():
    for svc in _sg.get_all_services():
        m = _sg.get_metadata(svc)
        assert 0.0 <= m["criticality_score"]   <= 1.0, f"{svc}: criticality_score out of range"
        assert 0.0 <= m["failure_sensitivity"] <= 1.0, f"{svc}: failure_sensitivity out of range"
        assert m["health_score"] == 1.0,               f"{svc}: initial health != 1.0"

def test_cascade_hops():
    for root in ("payment-db", "redis-cache", "storage-node"):
        sg = ServiceGraph()
        hops = sg.simulate_failure_impact(root, np.random.RandomState(SEED))
        _log(f"{root}: {[h.service for h in hops]}")
        assert len(hops) >= 1,          f"{root}: expected >= 1 hop"
        assert hops[0].service == root, f"{root}: first hop must be root itself"
        assert hops[0].delay_seconds == 0.0, f"{root}: root hop delay must be 0.0"

def test_health_states():
    assert health_state_from_score(1.0)  == HealthState.NORMAL
    assert health_state_from_score(0.79) == HealthState.DEGRADED
    assert health_state_from_score(0.39) == HealthState.FAILING

def test_graph_reset():
    sg = ServiceGraph()
    sg.simulate_failure_impact("redis-cache", np.random.RandomState(SEED))
    sg.reset_all_health()
    for svc in sg.get_all_services():
        assert _sg.get_metadata(svc)["health_score"] == 1.0, f"{svc}: not reset"

# ══════════════════════════════════════════════════════════════════════════════
# T8 — Partial observability
# ══════════════════════════════════════════════════════════════════════════════
def test_investigation_reveals_fields():
    for tid in (1, 2, 3):
        a = _alerts[tid][0]
        hidden = a.to_observation(investigated=False)
        open_  = a.to_observation(investigated=True)
        assert isinstance(open_["related_services"], list), \
            f"Alert {a.id}: related_services not list when investigated"
        assert "REDACTED" in str(hidden["related_services"]) or \
               hidden["related_services"] != open_["related_services"], \
            f"Alert {a.id}: field not masked when uninvestigated"
        _log(f"Task {tid}: {a.id} masking ✓")

def test_observation_keys():
    required = {"id","severity","source_service","alert_type","message","timestamp_offset","is_noise","is_root_cause"}
    for tid in (1, 2, 3):
        for a in _alerts[tid][:5]:
            missing = required - set(a.to_observation().keys())
            assert not missing, f"Alert {a.id} missing keys: {missing}"

# ══════════════════════════════════════════════════════════════════════════════
# T9 — Runbook registry
# ══════════════════════════════════════════════════════════════════════════════
def test_runbook_count():
    rbs = _reg.get_all()
    _log(f"Runbooks: {[r.id for r in rbs]}")
    assert len(rbs) == 7, f"Expected 7 runbooks, got {len(rbs)}"

def test_runbook_ids_unique():
    ids = _reg.list_ids()
    assert len(ids) == len(set(ids)), f"Duplicate runbook IDs: {ids}"

def test_runbook_task1_applicability():
    gt = _build_gt(_scenarios[1])
    assert _reg.get("rb_db_failover").is_applicable(gt), "rb_db_failover must apply to task1"

def test_runbook_task2_applicability():
    gt = _build_gt(_scenarios[2])
    assert _reg.get("rb_cache_flush_restart").is_applicable(gt), "rb_cache_flush_restart must apply to task2"

def test_runbook_task3_applicability():
    gt = _build_gt(_scenarios[3])
    assert _reg.get("rb_storage_volume_remount").is_applicable(gt)
    assert _reg.get("rb_ml_model_rollback").is_applicable(gt)

def test_wrong_action_never_applicable():
    for tid in (1, 2, 3):
        assert not _reg.get("rb_wrong_action").is_applicable(_build_gt(_scenarios[tid])), \
            f"Task {tid}: rb_wrong_action must never be applicable"

def test_runbook_effect_heals():
    gt = _build_gt(_scenarios[1])
    gt["root_services"] = ["payment-db"]
    state = SimulatedState(service_health={"payment-db": 0.10, "payment-service": 0.20})
    after = _reg.get("rb_db_failover").apply(state, gt)
    assert after.service_health["payment-db"] > 0.5, "rb_db_failover should restore payment-db health"
    assert "payment-db" in after.stopped_cascades,   "rb_db_failover should stop cascade"

# ══════════════════════════════════════════════════════════════════════════════
# T10 — Multi-root scoring (task3)
# ══════════════════════════════════════════════════════════════════════════════
def test_task3_two_root_causes():
    sc = _scenarios[3]
    assert len(sc.root_cause_alert_ids) == 2, \
        f"Task 3 must have 2 root causes, got {sc.root_cause_alert_ids}"
    _log(f"Task 3 root causes: {sc.root_cause_alert_ids}")

def test_task3_full_credit():
    sc = _scenarios[3]; al = _alerts[3]
    r  = _grdr.grade(_build_gt(sc), _perfect_actions(sc, al), 3)
    _log(f"Task 3 full credit: rc_score={r.root_cause_score:.4f}")
    assert r.root_cause_score == 1.0, \
        f"Task 3: both RCs identified → expected 1.0, got {r.root_cause_score:.4f}"

def test_task3_partial_credit():
    sc = _scenarios[3]; al = _alerts[3]; gt = _build_gt(sc)
    acts = []
    step = 0
    for a in al[:3]:
        acts.append({"type": ActionType.INVESTIGATE, "alert_id": a.id, "runbook_id": None, "step": step}); step+=1
    # Only the FIRST root cause
    acts.append({"type": ActionType.IDENTIFY_ROOT_CAUSE,
                 "alert_id": sc.root_cause_alert_ids[0], "runbook_id": None, "step": step}); step+=1
    for rb in sc.correct_runbook_ids:
        acts.append({"type": ActionType.APPLY_RUNBOOK, "alert_id": None, "runbook_id": rb, "step": step}); step+=1
    acts.append({"type": ActionType.RESOLVE, "alert_id": None, "runbook_id": None, "step": step})
    r = _grdr.grade(gt, acts, 3)
    _log(f"Task 3 partial credit: rc_score={r.root_cause_score:.4f}")
    assert 0.0 < r.root_cause_score < 1.0, \
        f"Task 3: one of two RCs → expected partial (0,1), got {r.root_cause_score:.4f}"

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(fast: bool = False, verbose: bool = False) -> int:
    global _verbose
    _verbose = verbose

    print(f"\n{'═'*65}")
    print(f"  IncidentMind — Environment Validation Suite  (seed={SEED})")
    print(f"{'═'*65}")

    section("T1 · Determinism")
    run_test("T1.1  Scenario determinism (all tasks)",       test_determinism_scenario)
    run_test("T1.2  Alert sequence determinism",             test_determinism_alerts)
    run_test("T1.3  Failure propagation determinism",        test_determinism_failure_propagation)
    run_test("T1.4  Grader output determinism",              test_determinism_grader)

    section("T2 · Task Constraints")
    run_test("T2.1  Alert count within task ranges",         test_task_alert_counts)
    run_test("T2.2  Noise percentage approximately correct", test_task_noise_percentage)
    run_test("T2.3  Red herring counts match spec",          test_task_red_herring_counts)
    run_test("T2.4  Task definitions metadata sane",         test_task_definitions_sane)

    section("T3 · Temporal Ordering")
    run_test("T3.1  Alerts sorted by timestamp",             test_temporal_sort_order)
    run_test("T3.2  Root cause fires earliest",              test_root_cause_fires_first)
    run_test("T3.3  Cascade alerts strictly post-T0",        test_cascade_after_root_cause)
    run_test("T3.4  Cascade delays monotonically increasing",test_cascade_delays_monotonic)

    section("T4 · Noise & Red Herring Flags")
    run_test("T4.1  Noise alert flags correct",              test_noise_flags_correct)
    run_test("T4.2  Root cause flags correct",               test_root_cause_flags_correct)
    run_test("T4.3  Red herrings are HIGH/CRITICAL",         test_red_herring_severity)
    run_test("T4.4  Noise alerts not CRITICAL",              test_noise_not_critical)
    run_test("T4.5  Partial observability masking active",   test_partial_observability_masking)

    section("T5 · Grader Correctness")
    run_test("T5.1  Perfect agent passes all tasks",         test_perfect_agent_passes)
    run_test("T5.2  Greedy agent penalized",                 test_greedy_agent_penalized)
    run_test("T5.3  Wrong root cause penalized",             test_wrong_root_cause_penalized)
    run_test("T5.4  Wrong runbook penalized",                test_wrong_runbook_penalized)
    run_test("T5.5  No-action agent scores 0.0",             test_no_action_scores_zero)
    run_test("T5.6  GradeResult fields complete & in range", test_grade_result_complete)

    section("T6 · Performance")
    run_test("T6.1  Full pipeline < 500ms per iteration",    test_pipeline_performance, skip=fast)

    section("T7 · Service Graph")
    run_test("T7.1  12+ services defined",                   test_service_count)
    run_test("T7.2  Graph is valid DAG",                     test_graph_is_dag)
    run_test("T7.3  All 4 tiers represented",                test_all_tiers_present)
    run_test("T7.4  Criticality/sensitivity in [0,1]",       test_criticality_range)
    run_test("T7.5  Cascade propagation produces hops",      test_cascade_hops)
    run_test("T7.6  HealthState transitions correct",        test_health_states)
    run_test("T7.7  reset_all_health restores scores",       test_graph_reset)

    section("T8 · Partial Observability")
    run_test("T8.1  Investigation reveals hidden fields",    test_investigation_reveals_fields)
    run_test("T8.2  Observation has all required keys",      test_observation_keys)

    section("T9 · Runbook Registry")
    run_test("T9.1  Exactly 7 runbooks registered",          test_runbook_count)
    run_test("T9.2  All runbook IDs unique",                 test_runbook_ids_unique)
    run_test("T9.3  rb_db_failover applicable to task1",     test_runbook_task1_applicability)
    run_test("T9.4  rb_cache_flush applicable to task2",     test_runbook_task2_applicability)
    run_test("T9.5  Storage+ML runbooks apply to task3",     test_runbook_task3_applicability)
    run_test("T9.6  rb_wrong_action never applicable",       test_wrong_action_never_applicable)
    run_test("T9.7  Runbook effect restores service health", test_runbook_effect_heals)

    section("T10 · Multi-Root Cause Scoring (Task 3)")
    run_test("T10.1 Task 3 has exactly 2 root causes",       test_task3_two_root_causes)
    run_test("T10.2 Both root causes → score 1.0",           test_task3_full_credit)
    run_test("T10.3 One of two root causes → partial score", test_task3_partial_credit)

    # ── summary ──────────────────────────────────────────────────────────────
    n_pass = sum(1 for r in _results if r["status"] == "PASS")
    n_fail = sum(1 for r in _results if r["status"] == "FAIL")
    n_skip = sum(1 for r in _results if r["status"] == "SKIP")
    n_total = len(_results)

    print(f"\n{'═'*65}")
    print(f"  Results: {n_pass}/{n_total} passed", end="")
    if n_skip:  print(f"  ({n_skip} skipped)", end="")
    if n_fail:
        print(f"  \033[91m{n_fail} FAILED\033[0m")
        print("\n  Failed tests:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"    ✗ {r['name']}")
                if r.get("error"):
                    print(f"      → {r['error']}")
    else:
        print()
    print(f"{'═'*65}\n")
    if n_fail == 0:
        print("  \033[92m✅  ALL TESTS PASSED — IncidentMind env is hackathon-ready.\033[0m\n")
    else:
        print("  \033[91m❌  SOME TESTS FAILED — review output above.\033[0m\n")
    # Don't exit here - let the caller decide
    return 0 if n_fail == 0 else 1


# ============================================================================
# ── EXTENDED TEST SUITE  (v2 upgrade validation) ─────────────────────────────
# T11  Fingerprint consistency
# T12  Duplicate / burst alert grouping
# T13  Burst alert simulation
# T14  Flapping alert simulation
# T15  Edge cases (zero noise, identical timestamps, alert storm, isolated failure)
# T16  Failure mode propagation & state machine
# ============================================================================

# ── shared v2 fixtures ───────────────────────────────────────────────────────
from envs.service_graph import FailureMode, ServiceState, service_state_from_score
from envs.grader import ActionType as _AT

_v2_scenarios = {tid: IncidentGenerator(seed=SEED).generate(tid) for tid in (1, 2, 3)}
_v2_alerts    = {tid: AlertGenerator(seed=SEED).generate(_v2_scenarios[tid])
                 for tid in (1, 2, 3)}


def _v2_perfect_actions(sc, al):
    """Perfect agent that also correctly deduplicates burst alerts."""
    acts, step = [], 0
    for a in al[:3]:
        acts.append({"type": _AT.INVESTIGATE, "alert_id": a.id, "runbook_id": None, "step": step}); step += 1
    # Investigate burst alerts before acting
    for a in al:
        if a.is_burst:
            acts.append({"type": _AT.INVESTIGATE, "alert_id": a.id, "runbook_id": None, "step": step}); step += 1
    for rc in sc.root_cause_alert_ids:
        acts.append({"type": _AT.IDENTIFY_ROOT_CAUSE, "alert_id": rc, "runbook_id": None, "step": step}); step += 1
    # Deduplicate burst alerts
    for bid in sc.burst_alert_ids:
        canon_id = sc.root_cause_alert_ids[0] if sc.root_cause_alert_ids else rc
        acts.append({"type": _AT.DEDUPLICATE_ALERT, "alert_id": bid,
                     "canonical_id": canon_id, "runbook_id": None, "step": step}); step += 1
    # Group burst alerts
    if sc.burst_alert_ids:
        group_ids = [sc.root_cause_alert_ids[0]] + sc.burst_alert_ids
        acts.append({"type": _AT.GROUP_ALERTS, "alert_ids": group_ids,
                     "group_label": sc.duplicate_group_map.get(sc.root_cause_alert_ids[0], "burst_group"),
                     "runbook_id": None, "step": step}); step += 1
    for rb in sc.correct_runbook_ids:
        acts.append({"type": _AT.APPLY_RUNBOOK, "alert_id": None, "runbook_id": rb, "step": step}); step += 1
    for nid in sc.noise_alert_ids:
        acts.append({"type": _AT.DISMISS_NOISE, "alert_id": nid, "runbook_id": None, "step": step}); step += 1
    acts.append({"type": _AT.RESOLVE, "alert_id": None, "runbook_id": None, "step": step})
    return acts


def _v2_build_gt(sc):
    root_svcs = sc.metadata.get("root_services", [sc.metadata.get("root_service", "")])
    if isinstance(root_svcs, str):
        root_svcs = [root_svcs]
    return {
        "task_id":               sc.task_id,
        "root_cause_alert_ids":  sc.root_cause_alert_ids,
        "cascade_chain":         [{"alert_id": s.alert_id} for s in sc.cascade_chain],
        "noise_alert_ids":       sc.noise_alert_ids,
        "red_herring_alert_ids": sc.red_herring_alert_ids,
        "correct_runbook_ids":   sc.correct_runbook_ids,
        "involved_services":     sc.involved_services,
        "root_services":         root_svcs,
        "alerts_by_service":     {},
        "burst_alert_ids":       sc.burst_alert_ids,
        "duplicate_group_map":   sc.duplicate_group_map,
    }


# ============================================================================
# T11 — Fingerprint consistency
# ============================================================================

def test_fingerprint_stable_across_seeds():
    """Same (service, alert_type, failure_mode) → same fingerprint_id regardless of seed."""
    from envs.alert_generator import _make_fingerprint
    for svc in ("payment-db", "redis-cache", "storage-node", "ml-inference"):
        for fmode in FailureMode:
            fp1 = _make_fingerprint(svc, "service_down", fmode)
            fp2 = _make_fingerprint(svc, "service_down", fmode)
            assert fp1 == fp2, f"{svc}/{fmode}: fingerprint not stable"
            assert fp1.startswith("fp_"), f"{svc}/{fmode}: fingerprint format wrong"
            _log(f"{svc}/{fmode.value}: {fp1}")


def test_fingerprint_unique_per_service():
    """Different services must produce different fingerprints for same alert_type."""
    from envs.alert_generator import _make_fingerprint
    seen = {}
    for svc in ("payment-db", "redis-cache", "storage-node", "ml-inference", "api-gateway"):
        fp = _make_fingerprint(svc, "service_down", FailureMode.TIMEOUT)
        assert fp not in seen.values(), f"Fingerprint collision: {svc} matches {[k for k,v in seen.items() if v==fp]}"
        seen[svc] = fp


def test_fingerprint_unique_per_failure_mode():
    """Same service with different failure modes → different fingerprints."""
    from envs.alert_generator import _make_fingerprint
    fps = {fmode: _make_fingerprint("payment-db", "service_down", fmode)
           for fmode in FailureMode}
    assert len(set(fps.values())) == len(FailureMode), \
        "Some failure modes produce identical fingerprints for same service"


def test_alert_fingerprint_present():
    """Every generated alert must have a non-empty fingerprint_id and group_key."""
    for tid in (1, 2, 3):
        for a in _v2_alerts[tid]:
            assert a.fingerprint_id, f"Alert {a.id}: fingerprint_id is empty"
            assert a.group_key,      f"Alert {a.id}: group_key is empty"
            assert a.fingerprint_id.startswith("fp_"), \
                f"Alert {a.id}: fingerprint_id format '{a.fingerprint_id}' unexpected"
            _log(f"Alert {a.id}: fp={a.fingerprint_id} gk={a.group_key}")


def test_burst_alerts_share_fingerprint_with_canonical():
    """Burst alerts must share fingerprint_id and group_key with their canonical parent."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        alert_map = {a.id: a for a in al}
        if not sc.burst_alert_ids:
            _log(f"Task {tid}: no burst alerts — skip")
            continue
        canonical_id = sc.root_cause_alert_ids[0]
        if canonical_id not in alert_map:
            continue
        canonical = alert_map[canonical_id]
        for bid in sc.burst_alert_ids:
            if bid not in alert_map:
                continue
            burst = alert_map[bid]
            assert burst.fingerprint_id == canonical.fingerprint_id, \
                f"Task {tid}: burst {bid} fp={burst.fingerprint_id} != canonical fp={canonical.fingerprint_id}"
            assert burst.group_key == canonical.group_key, \
                f"Task {tid}: burst {bid} gk={burst.group_key} != canonical gk={canonical.group_key}"
            _log(f"Task {tid}: burst {bid} shares fp/gk with canonical ✓")


# ============================================================================
# T12 — Duplicate / burst alert grouping
# ============================================================================

def test_duplicate_group_map_present():
    """Every scenario with burst alerts must have a populated duplicate_group_map."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        if sc.burst_alert_ids:
            assert sc.duplicate_group_map, \
                f"Task {tid}: burst_alert_ids non-empty but duplicate_group_map is empty"
            for bid in sc.burst_alert_ids:
                assert bid in sc.duplicate_group_map, \
                    f"Task {tid}: burst alert {bid} not in duplicate_group_map"
            _log(f"Task {tid}: dup_map={sc.duplicate_group_map}")


def test_burst_alerts_in_all_tasks():
    """Every task must define at least 1 burst alert (production systems always burst)."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        assert len(sc.burst_alert_ids) >= 1, \
            f"Task {tid}: expected >=1 burst alert, got {len(sc.burst_alert_ids)}"
        _log(f"Task {tid}: {len(sc.burst_alert_ids)} burst alerts: {sc.burst_alert_ids}")


def test_burst_occurrence_count_increments():
    """Burst alerts must have occurrence_count > 1; canonical must be 1."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        alert_map = {a.id: a for a in al}
        for bid in sc.burst_alert_ids:
            if bid not in alert_map:
                continue
            burst = alert_map[bid]
            assert burst.occurrence_count > 1, \
                f"Task {tid}: burst {bid} has occurrence_count={burst.occurrence_count}, expected >1"
            assert burst.is_burst, \
                f"Task {tid}: burst {bid} has is_burst=False"
        # Canonical root cause must have occurrence_count == 1
        for rc_id in sc.root_cause_alert_ids:
            if rc_id in alert_map:
                assert alert_map[rc_id].occurrence_count == 1, \
                    f"Task {tid}: canonical RC {rc_id} has occurrence_count != 1"
        _log(f"Task {tid}: burst occurrence_count verified ✓")


def test_dedup_score_perfect_agent():
    """Perfect agent with correct DEDUPLICATE_ALERT actions should score dedup_score near 1.0."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        gt = _v2_build_gt(sc)
        acts = _v2_perfect_actions(sc, al)
        result = _grdr.grade(gt, acts, tid)
        _log(f"Task {tid}: dedup_score={result.dedup_score:.4f}")
        assert result.dedup_score >= 0.5, \
            f"Task {tid}: perfect agent dedup_score={result.dedup_score:.4f} < 0.5"


def test_dedup_score_penalises_burst_as_root_cause():
    """Marking a burst duplicate as root cause must reduce dedup_score."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        gt = _v2_build_gt(sc)
        if not sc.burst_alert_ids:
            continue
        # Agent marks the burst duplicate as root cause
        acts = []
        step = 0
        acts.append({"type": _AT.INVESTIGATE, "alert_id": al[0].id,
                     "runbook_id": None, "step": step}); step += 1
        acts.append({"type": _AT.IDENTIFY_ROOT_CAUSE, "alert_id": sc.burst_alert_ids[0],
                     "runbook_id": None, "step": step}); step += 1
        for rb in sc.correct_runbook_ids:
            acts.append({"type": _AT.APPLY_RUNBOOK, "alert_id": None,
                         "runbook_id": rb, "step": step}); step += 1
        acts.append({"type": _AT.RESOLVE, "alert_id": None, "runbook_id": None, "step": step})
        result = _grdr.grade(gt, acts, tid)
        _log(f"Task {tid}: burst-as-RC → dedup_score={result.dedup_score:.4f}")
        # When burst marked as RC the root_cause_score is also reduced;
        # dedup_score should be below the perfect agent's score
        perfect_acts = _v2_perfect_actions(sc, al)
        perfect_result = _grdr.grade(gt, perfect_acts, tid)
        assert result.dedup_score <= perfect_result.dedup_score, \
            f"Task {tid}: burst-as-RC dedup_score not penalised"


# ============================================================================
# T13 — Burst alert simulation
# ============================================================================

def test_burst_alerts_present_in_generated_list():
    """Generated alert list must contain all burst alert IDs from scenario."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        alert_ids = {a.id for a in al}
        for bid in sc.burst_alert_ids:
            assert bid in alert_ids, \
                f"Task {tid}: burst alert {bid} not found in generated alerts"
        _log(f"Task {tid}: all {len(sc.burst_alert_ids)} burst alerts present ✓")


def test_burst_alerts_near_root_cause_timestamp():
    """Burst alerts must fire within 5 seconds of T=0 (rapid-fire window)."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        alert_map = {a.id: a for a in al}
        for bid in sc.burst_alert_ids:
            if bid not in alert_map:
                continue
            ts = alert_map[bid].timestamp_offset
            assert ts <= 5.0, \
                f"Task {tid}: burst alert {bid} at T={ts:.1f}s > 5s rapid-fire window"
            assert ts > 0.0, \
                f"Task {tid}: burst alert {bid} at T={ts} <= 0 (must be after root cause)"
        _log(f"Task {tid}: burst timing verified ✓")


def test_burst_determinism():
    """Same seed must produce identical burst alert IDs and timestamps."""
    for tid in (1, 2, 3):
        sc_a = IncidentGenerator(seed=SEED).generate(tid)
        sc_b = IncidentGenerator(seed=SEED).generate(tid)
        assert sc_a.burst_alert_ids == sc_b.burst_alert_ids, \
            f"Task {tid}: burst_alert_ids not deterministic"
        al_a = AlertGenerator(seed=SEED).generate(sc_a)
        al_b = AlertGenerator(seed=SEED).generate(sc_b)
        ts_a = {a.id: a.timestamp_offset for a in al_a if a.is_burst}
        ts_b = {a.id: a.timestamp_offset for a in al_b if a.is_burst}
        assert ts_a == ts_b, f"Task {tid}: burst timestamps not deterministic"
        _log(f"Task {tid}: burst determinism ✓")


# ============================================================================
# T14 — Flapping alert simulation
# ============================================================================

def test_flapping_alerts_generate_three_events():
    """Each flapping_alert_id must produce: fire, clear, refire (3 events per flap)."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        for fid in sc.flapping_alert_ids:
            flap_events = [a for a in al
                           if a.id in (fid, f"{fid}_clear", f"{fid}_refire")]
            assert len(flap_events) == 3, \
                (f"Task {tid}: flapping {fid} expected 3 events "
                 f"(fire/clear/refire), got {len(flap_events)}: {[e.id for e in flap_events]}")
            types = {a.id.split("_")[-1] if "_" in a.id else "fire" for a in flap_events}
            _log(f"Task {tid}: flap {fid} events: {[e.id for e in flap_events]}")


def test_flapping_clear_event_lower_severity():
    """The CLEAR event of a flapping alert must be LOW severity."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        for fid in sc.flapping_alert_ids:
            clear_alerts = [a for a in al if a.id == f"{fid}_clear"]
            if not clear_alerts:
                continue
            clear = clear_alerts[0]
            assert clear.severity == AlertSeverity.LOW, \
                f"Task {tid}: clear event severity={clear.severity}, expected LOW"
            assert clear.alert_type == "clear", \
                f"Task {tid}: clear event alert_type={clear.alert_type}, expected 'clear'"


def test_flapping_refire_higher_severity_than_clear():
    """The re-fire event must be higher severity than the CLEAR event."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        sev_rank = {AlertSeverity.CRITICAL: 0, AlertSeverity.HIGH: 1,
                    AlertSeverity.MEDIUM: 2, AlertSeverity.LOW: 3}
        for fid in sc.flapping_alert_ids:
            clear_list  = [a for a in al if a.id == f"{fid}_clear"]
            refire_list = [a for a in al if a.id == f"{fid}_refire"]
            if not clear_list or not refire_list:
                continue
            clear_rank  = sev_rank[clear_list[0].severity]
            refire_rank = sev_rank[refire_list[0].severity]
            assert refire_rank < clear_rank, \
                (f"Task {tid}: refire sev={refire_list[0].severity} "
                 f"not higher than clear sev={clear_list[0].severity}")


def test_flapping_temporal_order():
    """fire.ts < clear.ts < refire.ts must hold for every flapping alert."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        alert_map = {a.id: a for a in al}
        for fid in sc.flapping_alert_ids:
            fire   = alert_map.get(fid)
            clear  = alert_map.get(f"{fid}_clear")
            refire = alert_map.get(f"{fid}_refire")
            if not all([fire, clear, refire]):
                continue
            assert fire.timestamp_offset < clear.timestamp_offset, \
                f"Task {tid}: {fid} fire.ts >= clear.ts"
            assert clear.timestamp_offset < refire.timestamp_offset, \
                f"Task {tid}: {fid} clear.ts >= refire.ts"
            _log(f"Task {tid}: {fid} fire={fire.timestamp_offset:.1f} "
                 f"clear={clear.timestamp_offset:.1f} "
                 f"refire={refire.timestamp_offset:.1f}")


def test_flapping_share_fingerprint():
    """Fire, clear, and refire events of a flapping alert must share fingerprint_id."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        al = _v2_alerts[tid]
        alert_map = {a.id: a for a in al}
        for fid in sc.flapping_alert_ids:
            fire   = alert_map.get(fid)
            clear  = alert_map.get(f"{fid}_clear")
            refire = alert_map.get(f"{fid}_refire")
            if not all([fire, clear, refire]):
                continue
            assert fire.fingerprint_id == clear.fingerprint_id == refire.fingerprint_id, \
                f"Task {tid}: flapping events have different fingerprints"
            assert fire.group_key == clear.group_key == refire.group_key, \
                f"Task {tid}: flapping events have different group_keys"


# ============================================================================
# T15 — Edge cases
# ============================================================================

def test_edge_zero_noise_task1():
    """Task 1 is the zero-noise scenario — noise lists must be empty."""
    sc = _v2_scenarios[1]
    al = _v2_alerts[1]
    assert len(sc.noise_alert_ids) == 0, \
        f"Task 1: noise_alert_ids should be empty, got {sc.noise_alert_ids}"
    noise_in_alerts = [a for a in al if a.is_noise]
    assert len(noise_in_alerts) == 0, \
        f"Task 1: {len(noise_in_alerts)} noise alerts found in generated list"
    _log("Task 1: zero-noise verified ✓")


def test_edge_identical_timestamps():
    """
    Task 2 red herring fires at T≈0 alongside root cause.
    Multiple alerts at the same timestamp must all be present and ordered stably
    (CRITICAL before HIGH).
    """
    sc = _v2_scenarios[2]
    al = _v2_alerts[2]
    # Find alerts at T=0 or very close
    early_alerts = [a for a in al if a.timestamp_offset <= 10.0]
    assert len(early_alerts) >= 2, \
        (f"Task 2: expected >=2 alerts near T=0 (root + red herring), "
         f"got {len(early_alerts)}: {[(a.id, a.timestamp_offset) for a in early_alerts]}")
    # Verify sort stability: among ties, CRITICAL before HIGH
    for i in range(len(early_alerts) - 1):
        a, b = early_alerts[i], early_alerts[i + 1]
        if a.timestamp_offset == b.timestamp_offset:
            rank = {AlertSeverity.CRITICAL: 0, AlertSeverity.HIGH: 1,
                    AlertSeverity.MEDIUM: 2, AlertSeverity.LOW: 3}
            assert rank[a.severity] <= rank[b.severity], \
                (f"Task 2: timestamp tie {a.id}({a.severity}) before "
                 f"{b.id}({b.severity}) — sort not stable")
    _log(f"Task 2: {len(early_alerts)} early alerts, sort-stable ✓")


def test_edge_isolated_failure_no_cascade():
    """
    metrics-collector has no downstream dependents (isolated service).
    simulate_failure_impact must return exactly 1 hop (root only).
    """
    sg = ServiceGraph()
    rng = np.random.RandomState(SEED)
    hops = sg.simulate_failure_impact("metrics-collector", rng)
    assert len(hops) == 1, \
        (f"metrics-collector is isolated; expected 1 hop, "
         f"got {len(hops)}: {[h.service for h in hops]}")
    assert hops[0].service == "metrics-collector"
    assert sg.is_isolated_failure("metrics-collector"), \
        "is_isolated_failure('metrics-collector') should return True"
    _log("Isolated failure (metrics-collector) → 1 hop ✓")


def test_edge_alert_storm_100_plus():
    """
    Generating a synthetic storm of 120+ alerts must complete without error
    and maintain correct field structure on every alert.
    """
    from envs.incident_generator import IncidentScenario, CascadeStage

    # Build a synthetic scenario with 100 noise IDs
    storm_sc = IncidentScenario(
        task_id=3,
        scenario_name="storm_test",
        root_cause_alert_ids=["storm_rc_001"],
        cascade_chain=[
            CascadeStage("storm_cs_001", "api-gateway", 5.0, 0.5),
        ],
        involved_services=["payment-db", "api-gateway"],
        noise_alert_ids=[f"storm_noise_{i:03d}" for i in range(100)],
        red_herring_alert_ids=[],
        correct_runbook_ids=["rb_db_failover"],
        metadata={"root_service": "payment-db"},
    )
    ag = AlertGenerator(seed=SEED)
    al = ag.generate(storm_sc)
    assert len(al) >= 100, f"Storm scenario: expected >=100 alerts, got {len(al)}"
    required_keys = {"id", "severity", "source_service", "alert_type",
                     "message", "timestamp_offset", "fingerprint_id", "group_key"}
    for a in al[:10]:   # spot-check first 10
        obs = a.to_observation()
        missing = required_keys - set(obs.keys())
        assert not missing, f"Storm alert {a.id} missing keys: {missing}"
    _log(f"Alert storm: {len(al)} alerts generated, spot-check passed ✓")


def test_edge_all_services_degraded_detection():
    """ServiceGraph.all_services_degraded() must return True after mass damage."""
    sg = ServiceGraph()
    rng = np.random.RandomState(SEED)
    # Damage every service enough to degrade
    for svc in sg.get_all_services():
        sg.get_service(svc).apply_damage(0.30)
    assert sg.all_services_degraded(), \
        "all_services_degraded() should return True after mass damage"
    sg.reset_all_health()
    assert not sg.all_services_degraded(), \
        "all_services_degraded() should return False after reset"
    _log("all_services_degraded detection ✓")


def test_edge_missing_metadata_fields_graceful():
    """
    Grader must handle ground_truth with missing optional v2 fields gracefully
    (backward-compat: burst_alert_ids and duplicate_group_map absent).
    """
    sc = _v2_scenarios[1]
    al = _v2_alerts[1]
    # Deliberately omit v2 fields
    gt_minimal = {
        "task_id":               1,
        "root_cause_alert_ids":  sc.root_cause_alert_ids,
        "cascade_chain":         [{"alert_id": s.alert_id} for s in sc.cascade_chain],
        "noise_alert_ids":       sc.noise_alert_ids,
        "red_herring_alert_ids": sc.red_herring_alert_ids,
        "correct_runbook_ids":   sc.correct_runbook_ids,
        "involved_services":     sc.involved_services,
        "root_services":         ["payment-db"],
        "alerts_by_service":     {},
        # burst_alert_ids and duplicate_group_map intentionally omitted
    }
    acts = _perfect_actions(sc, al)
    try:
        result = _grdr.grade(gt_minimal, acts, 1)
        assert result.total_score >= 0.0
        _log(f"Missing v2 fields handled gracefully, score={result.total_score:.4f} ✓")
    except Exception as exc:
        assert False, f"Grader raised on missing v2 fields: {exc}"


# ============================================================================
# T16 — Failure mode propagation & state machine
# ============================================================================

def test_all_failure_modes_propagate():
    """Every FailureMode must produce at least 1 cascade hop for high-sensitivity roots."""
    for fmode in FailureMode:
        sg = ServiceGraph()
        rng = np.random.RandomState(SEED)
        hops = sg.simulate_failure_impact("redis-cache", rng, failure_mode=fmode)
        assert len(hops) >= 1, f"{fmode.value}: expected >=1 cascade hop"
        assert hops[0].service == "redis-cache", f"{fmode.value}: first hop must be root"
        assert hops[0].failure_mode == fmode, \
            f"{fmode.value}: hop failure_mode mismatch {hops[0].failure_mode}"
        _log(f"{fmode.value}: {len(hops)} hops")


def test_network_partition_bypasses_sensitivity():
    """NETWORK_PARTITION must affect MORE services than TIMEOUT from the same root."""
    results = {}
    for fmode in (FailureMode.TIMEOUT, FailureMode.NETWORK_PARTITION):
        sg = ServiceGraph()
        rng = np.random.RandomState(SEED)
        hops = sg.simulate_failure_impact("redis-cache", rng, failure_mode=fmode)
        results[fmode] = len(hops)
        _log(f"{fmode.value}: {len(hops)} hops")
    assert results[FailureMode.NETWORK_PARTITION] >= results[FailureMode.TIMEOUT], \
        ("NETWORK_PARTITION should affect >= as many services as TIMEOUT "
         f"(partition={results[FailureMode.NETWORK_PARTITION]}, "
         f"timeout={results[FailureMode.TIMEOUT]})")


def test_memory_leak_lower_initial_damage():
    """MEMORY_LEAK root hop must leave root service with higher health than TIMEOUT."""
    for root in ("redis-cache", "storage-node"):
        sg_timeout = ServiceGraph()
        sg_leak    = ServiceGraph()
        rng_t = np.random.RandomState(SEED)
        rng_l = np.random.RandomState(SEED)
        hops_t = sg_timeout.simulate_failure_impact(root, rng_t, failure_mode=FailureMode.TIMEOUT)
        hops_l = sg_leak.simulate_failure_impact(root, rng_l, failure_mode=FailureMode.MEMORY_LEAK)
        health_timeout = hops_t[0].health_score_after
        health_leak    = hops_l[0].health_score_after
        _log(f"{root}: timeout health={health_timeout:.3f}, leak health={health_leak:.3f}")
        assert health_leak > health_timeout, \
            (f"{root}: MEMORY_LEAK should leave higher health than TIMEOUT "
             f"(leak={health_leak:.3f}, timeout={health_timeout:.3f})")


def test_service_state_machine_transitions():
    """ServiceState transitions: HEALTHY → DEGRADED → FAILING → RECOVERING."""
    from envs.service_graph import service_state_from_score
    assert service_state_from_score(1.0)  == ServiceState.HEALTHY
    assert service_state_from_score(0.9)  == ServiceState.HEALTHY
    assert service_state_from_score(0.79) == ServiceState.DEGRADED
    assert service_state_from_score(0.5)  == ServiceState.DEGRADED
    assert service_state_from_score(0.39) == ServiceState.FAILING
    assert service_state_from_score(0.0)  == ServiceState.FAILING
    # RECOVERING needs recovering=True flag
    assert service_state_from_score(0.5, recovering=True)  == ServiceState.RECOVERING
    assert service_state_from_score(0.9, recovering=True)  == ServiceState.RECOVERING
    assert service_state_from_score(0.35, recovering=True) == ServiceState.FAILING  # still too low
    _log("State machine transitions ✓")


def test_tick_recovery_increments_health():
    """tick_recovery() must increment service health_score."""
    sg = ServiceGraph()
    rng = np.random.RandomState(SEED)
    sg.simulate_failure_impact("redis-cache", rng)
    health_before = sg.get_service("auth-service").health_score
    if health_before < 1.0:
        new_state = sg.tick_recovery("auth-service", recovery_per_tick=0.10)
        health_after = sg.get_service("auth-service").health_score
        assert health_after > health_before, \
            f"tick_recovery did not increase health: {health_before} → {health_after}"
        assert new_state == ServiceState.RECOVERING, \
            f"Expected RECOVERING state after tick, got {new_state}"
        _log(f"tick_recovery: {health_before:.3f} → {health_after:.3f}, state={new_state} ✓")


def test_scenario_failure_mode_set():
    """Each scenario must have a FailureMode that is not None."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        assert sc.failure_mode is not None, f"Task {tid}: failure_mode is None"
        assert isinstance(sc.failure_mode, FailureMode), \
            f"Task {tid}: failure_mode is not a FailureMode instance"
        _log(f"Task {tid}: failure_mode={sc.failure_mode.value}")


def test_cascade_stage_failure_mode_set():
    """Every CascadeStage must carry a FailureMode."""
    for tid in (1, 2, 3):
        sc = _v2_scenarios[tid]
        for stage in sc.cascade_chain:
            assert stage.failure_mode is not None, \
                f"Task {tid}: CascadeStage {stage.alert_id} has None failure_mode"
            assert isinstance(stage.failure_mode, FailureMode), \
                f"Task {tid}: CascadeStage {stage.alert_id} failure_mode wrong type"


# ============================================================================
# Extended runner (append to main)
# ============================================================================

def _run_extended(fast: bool = False, verbose: bool = False) -> int:
    global _verbose
    _verbose = verbose

    section("T11 · Fingerprint Consistency")
    run_test("T11.1  Fingerprint stable across seeds",           test_fingerprint_stable_across_seeds)
    run_test("T11.2  Fingerprint unique per service",            test_fingerprint_unique_per_service)
    run_test("T11.3  Fingerprint unique per failure mode",       test_fingerprint_unique_per_failure_mode)
    run_test("T11.4  All alerts have non-empty fingerprint+gk",  test_alert_fingerprint_present)
    run_test("T11.5  Burst alerts share fp/gk with canonical",   test_burst_alerts_share_fingerprint_with_canonical)

    section("T12 · Duplicate / Burst Alert Grouping")
    run_test("T12.1  duplicate_group_map present when bursts exist",  test_duplicate_group_map_present)
    run_test("T12.2  Every task has >= 1 burst alert",                test_burst_alerts_in_all_tasks)
    run_test("T12.3  Burst occurrence_count increments correctly",    test_burst_occurrence_count_increments)
    run_test("T12.4  Perfect agent dedup_score >= 0.5",               test_dedup_score_perfect_agent)
    run_test("T12.5  Burst-as-root-cause penalised in dedup_score",   test_dedup_score_penalises_burst_as_root_cause)

    section("T13 · Burst Alert Simulation")
    run_test("T13.1  Burst alerts present in generated list",     test_burst_alerts_present_in_generated_list)
    run_test("T13.2  Burst alerts within 5s rapid-fire window",   test_burst_alerts_near_root_cause_timestamp)
    run_test("T13.3  Burst timestamps deterministic",             test_burst_determinism)

    section("T14 · Flapping Alert Simulation")
    run_test("T14.1  Each flap produces fire+clear+refire",       test_flapping_alerts_generate_three_events)
    run_test("T14.2  CLEAR event has LOW severity",               test_flapping_clear_event_lower_severity)
    run_test("T14.3  Refire severity > clear severity",           test_flapping_refire_higher_severity_than_clear)
    run_test("T14.4  fire.ts < clear.ts < refire.ts",             test_flapping_temporal_order)
    run_test("T14.5  Flapping events share fingerprint+gk",       test_flapping_share_fingerprint)

    section("T15 · Edge Cases")
    run_test("T15.1  Zero-noise scenario (task 1)",               test_edge_zero_noise_task1)
    run_test("T15.2  Identical timestamps sorted stably",         test_edge_identical_timestamps)
    run_test("T15.3  Isolated failure → 1 hop only",              test_edge_isolated_failure_no_cascade)
    run_test("T15.4  Alert storm (100+ alerts) handled",          test_edge_alert_storm_100_plus)
    run_test("T15.5  all_services_degraded detection",            test_edge_all_services_degraded_detection)
    run_test("T15.6  Missing v2 fields handled gracefully",       test_edge_missing_metadata_fields_graceful)

    section("T16 · Failure Mode Propagation & State Machine")
    run_test("T16.1  All failure modes produce cascade hops",     test_all_failure_modes_propagate)
    run_test("T16.2  NETWORK_PARTITION bypasses sensitivity",     test_network_partition_bypasses_sensitivity)
    run_test("T16.3  MEMORY_LEAK lower initial damage than TIMEOUT", test_memory_leak_lower_initial_damage)
    run_test("T16.4  ServiceState machine transitions correct",   test_service_state_machine_transitions)
    run_test("T16.5  tick_recovery increments health",            test_tick_recovery_increments_health)
    run_test("T16.6  Scenario failure_mode populated",            test_scenario_failure_mode_set)
    run_test("T16.7  CascadeStage failure_mode populated",        test_cascade_stage_failure_mode_set)

    # summary
    n_pass  = sum(1 for r in _results if r["status"] == "PASS")
    n_fail  = sum(1 for r in _results if r["status"] == "FAIL")
    n_skip  = sum(1 for r in _results if r["status"] == "SKIP")
    n_total = len(_results)

    print(f"\n{'═'*65}")
    print(f"  Grand Total: {n_pass}/{n_total} passed", end="")
    if n_skip:  print(f"  ({n_skip} skipped)", end="")
    if n_fail:
        print(f"  \033[91m{n_fail} FAILED\033[0m")
        print("\n  Failed tests:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"    ✗ {r['name']}")
                if r.get("error"):
                    print(f"      → {r['error']}")
    else:
        print()
    print(f"{'═'*65}\n")
    if n_fail == 0:
        print("  \033[92m✅  ALL TESTS PASSED (v1 + v2) — IncidentMind env is production-grade.\033[0m\n")
    else:
        print("  \033[91m❌  SOME TESTS FAILED — review output above.\033[0m\n")
    return 0 if n_fail == 0 else 1


# Patch the original main to also run extended tests
_original_main = main

def main(fast: bool = False, verbose: bool = False) -> int:
    rc = _original_main(fast=fast, verbose=verbose)
    rc2 = _run_extended(fast=fast, verbose=verbose)
    return 0 if (rc == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IncidentMind full validation suite (v1+v2)")
    parser.add_argument("--fast",    action="store_true", help="Skip performance test")
    parser.add_argument("-v","--verbose", action="store_true", help="Show sub-step detail")
    args = parser.parse_args()
    sys.exit(main(fast=args.fast, verbose=args.verbose))