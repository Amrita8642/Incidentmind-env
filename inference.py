import os
import sys
import json
import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv("IncidentMind/.env")
except ImportError:
    pass

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional, used only if from_docker_image() is called

if HF_TOKEN is None:
    print("Error: HF_TOKEN environment variable is not set", flush=True)
    sys.exit(1)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an AI SRE assistant performing incident triage.
You will receive the current environment observation as JSON.
You must respond with ONLY a valid JSON object in this exact format:
{"action_type": "<type>", "parameters": {}}

Valid action_type values and their required parameters:
- INVESTIGATE: {"alert_id": "<id>"}
- MARK_ROOT_CAUSE: {"alert_id": "<id>"}
- GROUP_ALERTS: {"alert_ids": ["<id1>", "<id2>"]}
- SUPPRESS_ALERT: {"alert_id": "<id>"}
- TRIGGER_RUNBOOK: {"runbook_id": "<id>"}
- ESCALATE: {}
- RESOLVE: {}

Use the service_graph in the observation to trace cascades before marking root cause.
Do not include any explanation or markdown. Return only the JSON object."""

INCIDENT_SERVER_URL = "http://localhost:7860"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)

def run_inference():
    tasks = ["task1", "task2", "task3"]
    total_scores = []
    
    for task_id in tasks:
        log_start(task=task_id, env="IncidentMind", model=MODEL_NAME)
        
        try:
            response = requests.post(f"{INCIDENT_SERVER_URL}/reset", json={"task_id": task_id, "seed": 42})
            response.raise_for_status()
            obs = response.json()
        except Exception as e:
            print(f"Error calling /reset for {task_id}: {e}", flush=True)
            continue
            
        step_number = 0
        done = False
        total_score = 0.0
        success = False
        rewards = []
        
        while not done and step_number < 50:
            step_number += 1
            obs_str = json.dumps(obs)
            error_str = None
            
            try:
                llm_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_str}
                    ]
                )
                raw_text = llm_response.choices[0].message.content or ""
                
                try:
                    start_idx = raw_text.find('{')
                    end_idx = raw_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        action_json = json.loads(raw_text[start_idx:end_idx+1])
                    else:
                        action_json = json.loads(raw_text)
                except json.JSONDecodeError:
                    action_json = {"action_type": "RESOLVE", "parameters": {}}
                    error_str = "JSON parsing error"
                    
            except Exception as e:
                error_str = str(e)
                action_json = {"action_type": "RESOLVE", "parameters": {}}
            
            if not isinstance(action_json, dict) or "action_type" not in action_json or "parameters" not in action_json:
                action_json = {"action_type": "RESOLVE", "parameters": {}}
                error_str = error_str or "Invalid action format"
                
            action_type = action_json["action_type"]
            
            try:
                step_resp = requests.post(
                    f"{INCIDENT_SERVER_URL}/step", 
                    json={"action_type": action_type, "parameters": action_json["parameters"]}
                )
                step_resp.raise_for_status()
                step_result = step_resp.json()
            except Exception as e:
                error_str = str(e)
                done = True
                step_result = {"reward": 0.0, "done": True, "observation": obs, "info": {}}

            reward = float(step_result.get("reward", 0.0))
            rewards.append(reward)
            
            done = step_result.get("done", False)
            obs = step_result.get("observation", {})
            info = step_result.get("info", {})
            
            log_step(step=step_number, action=action_type, reward=reward, done=done, error=error_str)
            
            if done:
                grade = info.get("grade_result", info)
                if isinstance(grade, dict):
                    total_score = float(grade.get("total_score", 0.0))
                else:
                    total_score = 0.0
                success = total_score > 0.0
                
        log_end(success=success, steps=step_number, score=total_score, rewards=rewards)
        total_scores.append(total_score)
        
    if total_scores:
        avg_score = sum(total_scores) / len(total_scores)
        print(f"Average score: {avg_score:.4f}", flush=True)

if __name__ == "__main__":
    run_inference()
