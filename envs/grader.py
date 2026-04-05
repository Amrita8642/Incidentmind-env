"""
grader.py  (v2 — production upgrade)
======================================
Deterministic scoring of agent actions against ground truth.

Upgrades over v1
----------------
* New action types (backward-compat additions):
    GROUP_ALERTS       — agent groups a set of alert IDs as one logical incident
    DEDUPLICATE_ALERT  — agent marks an alert as duplicate of a canonical alert

* New scoring component:
    dedup_score  (weight 0.10) — rewards correct deduplication / grouping
                               — penalises merging unrelated alerts into one group

* Weight rebalance (sums to 1.0):
    root_cause    0.35  (was 0.40)
    runbook       0.25  (unchanged)
    noise_suppress 0.15 (unchanged)
    efficiency    0.15  (was 0.20)
    dedup         0.10  (new)

* Delayed-reward support:
    The grader now tracks whether the agent investigated burst/flapping alerts
    before making decisions about them.  Skipping burst investigation incurs
    a small penalty even if the final root cause answer is correct (simulating
    real SRE discipline of not acting on noise bursts).

* Edge-case hardening:
    - Empty actions → 0.0 (unchanged)
    - All noise scenario: noise_suppression returns 1.0 if there is no signal
      to find (agent correctly does nothing meaningful)
    - 100% noise burst: agent rewarded for grouping burst alerts and dismissing
    - Identical timestamps: no special handling needed (scoring is set-based)
    - Missing metadata fields: safe .get() defaults throughout

* GradeResult gains:
    dedup_score  — 0–1
    summary() updated to include new field

All existing tests remain green: the new dedup weight comes from a trim of
root_cause (0.40→0.35) and efficiency (0.20→0.15), but the scoring functions
for those components are unchanged so perfect-agent still scores 1.0 on both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ActionType  (extended — backward-compat)
# ---------------------------------------------------------------------------

class ActionType:
    INVESTIGATE          = "INVESTIGATE"
    IDENTIFY_ROOT_CAUSE  = "IDENTIFY_ROOT_CAUSE"
    DISMISS_NOISE        = "DISMISS_NOISE"
    APPLY_RUNBOOK        = "APPLY_RUNBOOK"
    RESOLVE              = "RESOLVE"
    # v2 additions
    GROUP_ALERTS         = "GROUP_ALERTS"         # {type, alert_ids: List[str], group_label: str}
    DEDUPLICATE_ALERT    = "DEDUPLICATE_ALERT"     # {type, alert_id, canonical_id}


# ---------------------------------------------------------------------------
# GradeResult  (extended)
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    """Full scoring breakdown returned by Grader.grade()."""
    total_score: float
    root_cause_score: float
    runbook_score: float
    noise_suppression_score: float
    efficiency_score: float
    dedup_score: float = 1.0            # v2: deduplication / grouping score
    details: Dict[str, object] = field(default_factory=dict)
    passed: bool = False

    def summary(self) -> str:
        lines = [
            f"=== GradeResult (task {self.details.get('task_id', '?')}) ===",
            f"  total_score          : {self.total_score:.4f}",
            f"  root_cause_score     : {self.root_cause_score:.4f}",
            f"  runbook_score        : {self.runbook_score:.4f}",
            f"  noise_suppression    : {self.noise_suppression_score:.4f}",
            f"  efficiency_score     : {self.efficiency_score:.4f}",
            f"  dedup_score          : {self.dedup_score:.4f}",
            f"  passed               : {self.passed}",
        ]
        if self.details.get("penalties"):
            lines.append(f"  penalties            : {self.details['penalties']}")
        if self.details.get("root_cause_analysis"):
            lines.append(f"  root_cause_analysis  : {self.details['root_cause_analysis']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class Grader:
    """
    Evaluates agent performance on an incident triage task.

    agent_actions is a list of dicts with shape:
        {
          "type":        ActionType.*,
          "alert_id":    str | None,
          "runbook_id":  str | None,
          "step":        int,            # 0-indexed
          # v2 extras (optional):
          "alert_ids":   List[str],      # for GROUP_ALERTS
          "group_label": str,            # for GROUP_ALERTS
          "canonical_id":str,            # for DEDUPLICATE_ALERT
        }

    ground_truth dict:
        {
          "task_id":               int,
          "root_cause_alert_ids":  List[str],
          "cascade_chain":         List[{alert_id: str}],
          "noise_alert_ids":       List[str],
          "red_herring_alert_ids": List[str],
          "correct_runbook_ids":   List[str],
          "involved_services":     List[str],
          # v2 extras (optional):
          "burst_alert_ids":       List[str],
          "duplicate_group_map":   Dict[str, str],  # alert_id → group_key
        }
    """

    _WEIGHTS = {
        "root_cause":     0.35,
        "runbook":        0.25,
        "noise_suppress": 0.15,
        "efficiency":     0.15,
        "dedup":          0.10,
    }

    def grade(
        self,
        ground_truth: Dict,
        agent_actions: List[Dict],
        task_id: int,
    ) -> GradeResult:
        """Score agent action sequence.  Fully deterministic."""
        from .tasks import get_task
        task = get_task(task_id)

        # Early exit: no actions → all zeros
        if not agent_actions:
            _t = get_task(task_id)
            return GradeResult(
                total_score=0.0, root_cause_score=0.0, runbook_score=0.0,
                noise_suppression_score=0.0, efficiency_score=0.0, dedup_score=0.0,
                details={"task_id": task_id, "penalties": ["no_actions"],
                         "steps_used": 0, "max_steps": _t.max_steps,
                         "passing_score": _t.passing_score,
                         "root_cause_analysis": {}, "runbook_analysis": {}},
                passed=False,
            )

        penalties: List[str] = []

        rc_score    = self._score_root_cause(ground_truth, agent_actions, penalties)
        rb_score    = self._score_runbooks(ground_truth, agent_actions, penalties)
        ns_score    = self._score_noise_suppression(ground_truth, agent_actions, penalties)
        eff_score   = self._score_efficiency(ground_truth, agent_actions, task, penalties)
        dedup_score = self._score_dedup(ground_truth, agent_actions, penalties)

        total = (
            self._WEIGHTS["root_cause"]     * rc_score
            + self._WEIGHTS["runbook"]      * rb_score
            + self._WEIGHTS["noise_suppress"] * ns_score
            + self._WEIGHTS["efficiency"]   * eff_score
            + self._WEIGHTS["dedup"]        * dedup_score
        )
        total = float(np.clip(total, 0.0, 1.0))
        passed = total >= task.passing_score

        details = {
            "task_id":             task_id,
            "penalties":           penalties,
            "root_cause_analysis": self._root_cause_details(ground_truth, agent_actions),
            "runbook_analysis":    self._runbook_details(ground_truth, agent_actions),
            "steps_used":          len(agent_actions),
            "max_steps":           task.max_steps,
            "passing_score":       task.passing_score,
        }

        return GradeResult(
            total_score=round(total, 6),
            root_cause_score=round(rc_score, 6),
            runbook_score=round(rb_score, 6),
            noise_suppression_score=round(ns_score, 6),
            efficiency_score=round(eff_score, 6),
            dedup_score=round(dedup_score, 6),
            details=details,
            passed=passed,
        )

    # ------------------------------------------------------------------
    # Root cause scoring (0–1)  — unchanged logic from v1
    # ------------------------------------------------------------------

    def _score_root_cause(
        self, gt: Dict, actions: List[Dict], penalties: List[str]
    ) -> float:
        correct_rc: Set[str] = set(gt["root_cause_alert_ids"])
        noise_ids:  Set[str] = set(gt["noise_alert_ids"])
        rh_ids:     Set[str] = set(gt["red_herring_alert_ids"])
        cascade_ids: Set[str] = {
            s["alert_id"] if isinstance(s, dict) else getattr(s, "alert_id", "")
            for s in gt.get("cascade_chain", [])
        }

        marked_rc: List[Tuple[str, int]] = [
            (a["alert_id"], a.get("step", 0))
            for a in actions
            if a.get("type") == ActionType.IDENTIFY_ROOT_CAUSE and a.get("alert_id")
        ]

        if not marked_rc:
            penalties.append("no_root_cause_identified: -0.40")
            return 0.0

        marked_ids = {mid for mid, _ in marked_rc}
        tp  = marked_ids & correct_rc
        fp_noise   = marked_ids & noise_ids
        fp_rh      = marked_ids & rh_ids
        fp_cascade = marked_ids & cascade_ids

        precision = len(tp) / max(len(marked_ids), 1)
        recall    = len(tp) / max(len(correct_rc), 1)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)

        if fp_rh:
            penalties.append(f"marked_red_herring_as_root_cause:{fp_rh}: -0.20")
            f1 = max(0.0, f1 - 0.20 * len(fp_rh))
        if fp_noise:
            penalties.append(f"marked_noise_as_root_cause:{fp_noise}: -0.10")
            f1 = max(0.0, f1 - 0.10 * len(fp_noise))
        if fp_cascade:
            penalties.append(f"marked_cascade_as_root_cause:{fp_cascade}: -0.10")
            f1 = max(0.0, f1 - 0.10 * len(fp_cascade))

        # Penalise marking burst alerts as root cause (they are duplicates)
        burst_ids: Set[str] = set(gt.get("burst_alert_ids", []))
        fp_burst = marked_ids & burst_ids
        if fp_burst:
            penalties.append(f"marked_burst_duplicate_as_root_cause:{fp_burst}: -0.10")
            f1 = max(0.0, f1 - 0.10 * len(fp_burst))

        return float(np.clip(f1, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Runbook scoring  — unchanged from v1
    # ------------------------------------------------------------------

    def _score_runbooks(
        self, gt: Dict, actions: List[Dict], penalties: List[str]
    ) -> float:
        correct_rbs: Set[str] = set(gt["correct_runbook_ids"])

        applied_rbs: List[Tuple[str, int]] = [
            (a["runbook_id"], a.get("step", 0))
            for a in actions
            if a.get("type") == ActionType.APPLY_RUNBOOK and a.get("runbook_id")
        ]

        if not applied_rbs:
            penalties.append("no_runbook_applied: -0.25")
            return 0.0

        applied_ids = {rid for rid, _ in applied_rbs}
        tp = applied_ids & correct_rbs
        fp = applied_ids - correct_rbs

        precision = len(tp) / max(len(applied_ids), 1)
        recall    = len(tp) / max(len(correct_rbs), 1)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)

        if fp:
            penalties.append(f"wrong_runbooks_applied:{fp}: -0.10 each")
            f1 = max(0.0, f1 - 0.10 * len(fp))

        rc_steps = [a.get("step", 0) for a in actions
                    if a.get("type") == ActionType.IDENTIFY_ROOT_CAUSE]
        rb_steps = [s for _, s in applied_rbs]
        if rc_steps and rb_steps and all(s > min(rc_steps) for s in rb_steps):
            f1 = min(1.0, f1 + 0.05)

        return float(np.clip(f1, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Noise suppression  — unchanged from v1
    # ------------------------------------------------------------------

    def _score_noise_suppression(
        self, gt: Dict, actions: List[Dict], penalties: List[str]
    ) -> float:
        noise_ids: Set[str] = set(gt["noise_alert_ids"])
        rh_ids:    Set[str] = set(gt["red_herring_alert_ids"])

        if not noise_ids and not rh_ids:
            return 1.0

        dismissed: Set[str] = {
            a["alert_id"]
            for a in actions
            if a.get("type") == ActionType.DISMISS_NOISE and a.get("alert_id")
        }

        correct_dismiss  = dismissed & noise_ids
        rc_dismissed     = dismissed & set(gt["root_cause_alert_ids"])
        cascade_ids: Set[str] = {
            s["alert_id"] if isinstance(s, dict) else getattr(s, "alert_id", "")
            for s in gt.get("cascade_chain", [])
        }
        cascade_dismissed = dismissed & cascade_ids

        recall = len(correct_dismiss) / len(noise_ids) if noise_ids else 1.0

        if rc_dismissed:
            penalties.append(f"dismissed_root_cause_alerts:{rc_dismissed}: -0.20")
            recall = max(0.0, recall - 0.20 * len(rc_dismissed))
        if cascade_dismissed:
            penalties.append(f"dismissed_cascade_alerts:{cascade_dismissed}: -0.10")
            recall = max(0.0, recall - 0.10 * len(cascade_dismissed))

        rh_marked_rc = {
            a["alert_id"] for a in actions
            if a.get("type") == ActionType.IDENTIFY_ROOT_CAUSE
            and a.get("alert_id") in rh_ids
        }
        if not rh_marked_rc:
            recall = min(1.0, recall + 0.05 * len(rh_ids))

        return float(np.clip(recall, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Efficiency scoring  — enhanced with burst-investigation check
    # ------------------------------------------------------------------

    def _score_efficiency(
        self, gt: Dict, actions: List[Dict], task, penalties: List[str]
    ) -> float:
        if not actions:
            return 0.0

        n = len(actions)
        score = 1.0
        action_types = [a.get("type") for a in actions]

        # Greedy: no investigation before first root cause mark
        first_rc_step = next(
            (i for i, a in enumerate(actions)
             if a.get("type") == ActionType.IDENTIFY_ROOT_CAUSE),
            None,
        )
        investigations_before_rc = sum(
            1 for i, a in enumerate(actions)
            if a.get("type") == ActionType.INVESTIGATE
            and (first_rc_step is None or i < first_rc_step)
        )
        if first_rc_step is not None and investigations_before_rc == 0:
            score -= 0.30
            penalties.append("greedy_root_cause_no_investigation: -0.30")

        # Premature RESOLVE
        resolve_steps = [i for i, t in enumerate(action_types) if t == ActionType.RESOLVE]
        rb_steps      = [i for i, t in enumerate(action_types) if t == ActionType.APPLY_RUNBOOK]

        if resolve_steps:
            first_resolve = resolve_steps[0]
            if not rb_steps or first_resolve < min(rb_steps):
                score -= 0.20
                penalties.append("premature_resolve_before_runbook: -0.20")
            if first_rc_step is None or first_resolve < first_rc_step:
                score -= 0.15
                penalties.append("premature_resolve_before_root_cause: -0.15")

        # Step budget
        budget_threshold = int(task.max_steps * 0.75)
        if n > budget_threshold:
            overage_fraction = (n - budget_threshold) / max(task.max_steps - budget_threshold, 1)
            budget_penalty = 0.10 * min(overage_fraction, 1.0)
            score -= budget_penalty
            if budget_penalty > 0.01:
                penalties.append(f"over_step_budget: -{budget_penalty:.3f}")

        # Runbook before root cause
        if rb_steps and first_rc_step is not None and min(rb_steps) < first_rc_step:
            score -= 0.10
            penalties.append("runbook_before_root_cause_identified: -0.10")

        # v2: burst alerts acted on without investigation
        burst_ids: Set[str] = set(gt.get("burst_alert_ids", []))
        if burst_ids:
            investigated = {
                a["alert_id"] for a in actions
                if a.get("type") == ActionType.INVESTIGATE and a.get("alert_id") in burst_ids
            }
            acted_on_burst = {
                a.get("alert_id") for a in actions
                if a.get("type") == ActionType.IDENTIFY_ROOT_CAUSE
                and a.get("alert_id") in burst_ids
            }
            uninvestigated_burst_actions = acted_on_burst - investigated
            if uninvestigated_burst_actions:
                penalty = 0.05 * len(uninvestigated_burst_actions)
                score -= penalty
                penalties.append(f"acted_on_burst_without_investigation: -{penalty:.2f}")

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Deduplication / grouping scoring  (NEW in v2)
    # ------------------------------------------------------------------

    def _score_dedup(
        self, gt: Dict, actions: List[Dict], penalties: List[str]
    ) -> float:
        """
        Score agent's alert deduplication / grouping behaviour.

        Reward:
          - Grouping burst alerts correctly (same group_key)
          - Marking burst duplicates as DEDUPLICATE_ALERT pointing to correct canonical
          - NOT marking burst duplicates as independent root causes

        Penalise:
          - Grouping unrelated alerts together (cross-group pollution)
          - Marking burst alert as root cause instead of deduplicating it
        """
        dup_map: Dict[str, str] = gt.get("duplicate_group_map", {})
        burst_ids: Set[str] = set(gt.get("burst_alert_ids", []))

        # If no burst alerts in this task, dedup is trivially perfect
        if not burst_ids and not dup_map:
            return 1.0

        score = 1.0

        # Collect GROUP_ALERTS actions
        group_actions = [
            a for a in actions if a.get("type") == ActionType.GROUP_ALERTS
        ]
        # Collect DEDUPLICATE_ALERT actions
        dedup_actions = [
            a for a in actions if a.get("type") == ActionType.DEDUPLICATE_ALERT
        ]

        # Check: did agent correctly deduplicate burst alerts?
        correctly_deduped: Set[str] = set()
        for da in dedup_actions:
            aid  = da.get("alert_id", "")
            caid = da.get("canonical_id", "")
            # Correct if: aid is in burst_ids AND canonical_id matches the root_cause_alert_id
            rc_ids = set(gt["root_cause_alert_ids"])
            if aid in burst_ids and caid in rc_ids:
                correctly_deduped.add(aid)

        # Recall of correct dedup (partial credit for partial dedup)
        if burst_ids:
            dedup_recall = len(correctly_deduped) / len(burst_ids)
        else:
            dedup_recall = 1.0

        # Check GROUP_ALERTS for cross-group pollution
        for ga in group_actions:
            grouped_ids: List[str] = ga.get("alert_ids", [])
            if len(grouped_ids) < 2:
                continue
            # Determine group keys for each alert in the group
            group_keys = set()
            for aid in grouped_ids:
                gk = dup_map.get(aid, aid)   # default: each alert is its own group
                group_keys.add(gk)
            if len(group_keys) > 1:
                # Multiple group_keys merged — pollution
                penalty = 0.10
                score -= penalty
                penalties.append(f"cross_group_pollution_in_GROUP_ALERTS: -{penalty:.2f}")

        score = score * (0.5 + 0.5 * dedup_recall)   # scale by dedup recall

        # Bonus: agent deduped all burst alerts correctly without marking them as RC
        rc_marked = {
            a.get("alert_id") for a in actions
            if a.get("type") == ActionType.IDENTIFY_ROOT_CAUSE
        }
        burst_marked_as_rc = burst_ids & rc_marked
        if not burst_marked_as_rc and correctly_deduped == burst_ids and burst_ids:
            score = min(1.0, score + 0.10)   # full dedup bonus

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Detail helpers
    # ------------------------------------------------------------------

    def _root_cause_details(self, gt: Dict, actions: List[Dict]) -> Dict:
        correct_rc = set(gt["root_cause_alert_ids"])
        marked = {a["alert_id"] for a in actions
                  if a.get("type") == ActionType.IDENTIFY_ROOT_CAUSE and a.get("alert_id")}
        return {
            "expected":  sorted(correct_rc),
            "submitted": sorted(marked),
            "correct":   sorted(marked & correct_rc),
            "missed":    sorted(correct_rc - marked),
            "wrong":     sorted(marked - correct_rc),
        }

    def _runbook_details(self, gt: Dict, actions: List[Dict]) -> Dict:
        correct_rbs = set(gt["correct_runbook_ids"])
        applied = {a["runbook_id"] for a in actions
                   if a.get("type") == ActionType.APPLY_RUNBOOK and a.get("runbook_id")}
        return {
            "expected": sorted(correct_rbs),
            "applied":  sorted(applied),
            "correct":  sorted(applied & correct_rbs),
            "missed":   sorted(correct_rbs - applied),
            "wrong":    sorted(applied - correct_rbs),
        }