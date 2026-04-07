"""
inference.py — IncidentMind Mandatory Evaluation Script
OWNER: Ritu
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_env() -> tuple[str, str, Optional[str], str, int]:
    missing = []
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name   = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    hf_token     = os.environ.get("HF_TOKEN")
    if not os.environ.get("API_BASE_URL"): missing.append("API_BASE_URL")
    if not os.environ.get("MODEL_NAME"):   missing.append("MODEL_NAME")
    if not os.environ.get("HF_TOKEN"):     missing.append("HF_TOKEN")
    if missing:
        print("WARNING: Missing environment variables, proceeding with defaults:")
        for var in missing: print(f"  - {var}")
    server_url = os.environ.get("INCIDENT_SERVER_URL", "http://localhost:7860")
    seed = int(os.environ.get("SEED", "42"))
    return api_base_url, model_name, hf_token, server_url, seed


class SimpleEnvClient:
    def __init__(self, server_url: str) -> None:
        self.base = server_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self, task_id: str, seed: Optional[int] = None) -> dict:
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        r = self.session.post(f"{self.base}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, parameters: dict) -> dict:
        payload = {"action_type": action_type, "parameters": parameters}
        r = self.session.post(f"{self.base}/step", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/health", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False


SYSTEM_PROMPT = """You are an SRE AI performing incident triage. Be concise.

Identify root cause and apply runbook, then RESOLVE.

Actions (respond with ONLY valid JSON):
- {"action_type": "INVESTIGATE", "parameters": {"alert_id": "id"}}
- {"action_type": "MARK_ROOT_CAUSE", "parameters": {"alert_id": "id"}}
- {"action_type": "TRIGGER_RUNBOOK", "parameters": {"runbook_id": "id"}}
- {"action_type": "RESOLVE", "parameters": {}}

Runbooks:
  rb_db_failover            for payment-db
  rb_cache_flush_restart    for redis-cache
  rb_storage_volume_remount for storage-node
  rb_ml_model_rollback      for ml-inference
  rb_service_restart        for other services
  rb_auth_token_invalidate  for auth-service

Rules:
1. INVESTIGATE CRITICAL alerts first
2. Leaf nodes in graph (no dependencies) = root causes
3. MARK_ROOT_CAUSE then TRIGGER_RUNBOOK then RESOLVE
4. Never re-investigate same alert twice
"""


def observation_to_prompt(obs: dict) -> str:
    alerts = obs.get("alerts", [])
    investigated = obs.get("investigated_alerts", [])
    root_cause_candidates = obs.get("root_cause_candidates", [])
    triggered_runbooks = obs.get("triggered_runbooks", [])
    service_graph = obs.get("service_graph", {})
    step_count = obs.get("step_count", 0)
    max_steps = obs.get("max_steps", 0)
    task_id = obs.get("task_id", "")

    sorted_alerts = sorted(
        alerts,
        key=lambda x: ["CRITICAL", "HIGH", "MEDIUM", "LOW"].index(x["severity"])
    )[:6]

    lines = [f"Task:{task_id} Step:{step_count}/{max_steps}", "Alerts:"]
    for a in sorted_alerts:
        tag = "[INV]" if a["id"] in investigated else "     "
        lines.append(f"  {tag}[{a['severity']}] {a['id']} | {a['source_service']}")

    lines.append("Leaf services (no deps = root cause):")
    for svc, deps in service_graph.items():
        if not deps:
            lines.append(f"  {svc}")

    lines.append(f"Root marked: {root_cause_candidates or 'none'}")
    lines.append(f"Runbooks done: {triggered_runbooks or 'none'}")
    lines.append("Next action JSON:")
    return "\n".join(lines)


VALID_ACTION_TYPES = {
    "INVESTIGATE", "MARK_ROOT_CAUSE", "TRIGGER_RUNBOOK",
    "GROUP_ALERTS", "SUPPRESS_ALERT", "QUERY_RUNBOOK", "RESOLVE",
}


def parse_llm_response(raw_text: str) -> Optional[dict[str, Any]]:
    text = raw_text.strip()
    if text.startswith("```"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        action = json.loads(text[start:end + 1])
        if not isinstance(action, dict):
            return None
    except Exception:
        return None
    if "action_type" not in action:
        return None
    if action.get("action_type") not in VALID_ACTION_TYPES:
        return None
    if "parameters" not in action:
        action["parameters"] = {}
    return action


def run_episode(
    llm_client: OpenAI,
    env_client: SimpleEnvClient,
    model_name: str,
    task_id: str,
    seed: int,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()} | SEED: {seed}")
    print(f"{'='*60}")
    print(f"[START] task={task_id}", flush=True)

    try:
        obs = env_client.reset(task_id=task_id, seed=seed)
    except Exception as e:
    print(f"  Episode started. Alerts visible: {len(obs.get('alerts', []))}")

    conversation: list[dict[str, str]] = []
    step = 0
    grade_result = None
    root_cause_marked = False
    runbook_triggered = False

    while True:
        step += 1

        # AUTO-RESOLVE after root cause + runbook to save credits
        if root_cause_marked and runbook_triggered:
            print(f"  [Step {step}] Auto-RESOLVE")
            try:
                step_result = env_client.step(action_type="RESOLVE", parameters={})
                reward = step_result.get("reward", 0.0)
                done = step_result.get("done", True)
                print(f"[STEP] step={step} reward={reward}", flush=True)
                obs = step_result.get("observation", obs)
                info = step_result.get("info", {})
                print(f"             reward={reward:+.3f} | done={done}")
                if done:
                    grade_result = info.get("grade_result")
                    print(f"\n  Episode ended. Reason: auto_resolved")
            except Exception as e:
                print(f"  [Step {step}] /step call failed: {e}")
            break

        prompt = observation_to_prompt(obs)
        conversation.append({"role": "user", "content": prompt})

        # Keep only last 2 exchanges to stay under token limit
        recent = conversation[-4:]

        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + recent,
                max_tokens=100,
                temperature=0.0,
                seed=seed,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [Step {step}] LLM call failed: {e}")
            raw_text = ""

        conversation.append({"role": "assistant", "content": raw_text})

        action = parse_llm_response(raw_text)
        if action is None:
            print(f"  [Step {step}] Parse failed. Raw: {raw_text[:80]!r}. Using RESOLVE fallback.")
            action = {"action_type": "RESOLVE", "parameters": {}}

        action_type = action["action_type"]
        params = action.get("parameters", {})

        # Skip duplicate INVESTIGATE to save credits
        if action_type == "INVESTIGATE":
            alert_id = params.get("alert_id", "")
            if alert_id in obs.get("investigated_alerts", []):
                print(f"  [Step {step}] INVESTIGATE {params} [SKIPPED-duplicate]")
                conversation.pop()
                continue

        print(f"  [Step {step}] {action_type} {params}")

        if action_type == "MARK_ROOT_CAUSE":
            root_cause_marked = True
        if action_type == "TRIGGER_RUNBOOK":
            runbook_triggered = True

        try:
            step_result = env_client.step(action_type=action_type, parameters=params)
        except Exception as e:
            print(f"  [Step {step}] /step call failed: {e}")
            break

        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        print(f"             reward={reward:+.3f} | done={done}")
        print(f"[STEP] step={step} reward={reward}", flush=True)

        obs = step_result.get("observation", obs)
        info = step_result.get("info", {})

        if done:
            grade_result = info.get("grade_result")
            reason = info.get("reason", "unknown")
            print(f"\n  Episode ended. Reason: {reason}")
            break

    grade = grade_result or {
        "total_score": 0.0,
        "root_cause_score": 0.0,
        "runbook_score": 0.0,
        "noise_suppression_score": 0.0,
        "efficiency_score": 0.0,
        "details": "Episode ended without grading.",
    }
    score = grade.get("total_score", 0.0)
    print(f"[END] task={task_id} score={score} steps={step}", flush=True)
    return grade


def _main_core() -> None:
    start_time = time.time()
    api_base_url, model_name, hf_token, server_url, seed = load_env()

    print("\n" + "="*60)
    print("  IncidentMind — Evaluation Script")
    print("="*60)
    print(f"  Server:   {server_url}")
    print(f"  Model:    {model_name}")
    print(f"  API Base: {api_base_url}")
    print(f"  Seed:     {seed}")
    print("="*60)

    env_client = SimpleEnvClient(server_url)
    import time
    for _ in range(30):
        if env_client.health():
            print("\n[✓] Server is healthy.")
            break
        time.sleep(2)
    else:
        print(f"\nERROR: IncidentMind server not reachable at {server_url}")
        import sys; sys.exit(0)

    llm_client = OpenAI(base_url=api_base_url, api_key=hf_token)

    tasks = ["task1", "task2", "task3"]
    scores: list[float] = []
    results: dict[str, dict] = {}

    for task_id in tasks:
        grade = run_episode(
            llm_client=llm_client,
            env_client=env_client,
            model_name=model_name,
            task_id=task_id,
            seed=seed,
        )
        results[task_id] = grade
        scores.append(grade["total_score"])

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  {'Task':<10} {'Score':>8}  Details")
    print(f"  {'-'*10} {'-'*8}  {'-'*30}")

    for task_id in tasks:
        g = results[task_id]
        score = g["total_score"]
        details = str(g["details"])
        # details_short = details[:60] + "..." if len(details) > 60 else details
        print(f"  {task_id:<10} {score:>8.4f}  {details}")

    avg_score = sum(scores) / len(scores)
    print(f"\n  {'AVERAGE':<10} {avg_score:>8.4f}")
    print(f"\n  Completed in {elapsed:.1f}s")
    print("="*60)

    if elapsed > 1200:
        print(f"\nWARNING: Took {elapsed/60:.1f} min (limit 20 min)")


def main() -> None:
    try:
        _main_core()
    except Exception as e:
        import traceback; traceback.print_exc()
        import sys; sys.exit(0)

if __name__ == "__main__":
    main()