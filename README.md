---
title: IncidentMind
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# IncidentMind 🚀

An AI-powered SRE (Site Reliability Engineering) incident triage environment built for the **Meta PyTorch × Hugging Face OpenEnv Hackathon 2026**.

IncidentMind trains and evaluates AI agents to identify root causes, trigger runbooks, and resolve incidents across multiple difficulty tiers (EASY, MEDIUM, HARD). It simulates realistic microservice failures, cascading alerts, and noisy environments.

## 📖 Overview

This project provides a fully interactive environment where an LLM (agent) receives server system alerts, service graph topologies, and incident metadata. The agent must investigate alerts, mark the root cause, and apply the correct runbook to resolve the simulated outage.

The core HTTP API is powered by FastAPI, and the environment state is strictly managed using Pydantic v2 models.

## 📂 Project Structure

```text
incidentmind/
├── __init__.py                    # Package exports
├── models.py                      # Pydantic v2 models (Observation, Action, etc.)
├── client.py                      # IncidentEnvClient (async + sync wrappers)
├── inference.py                   # Mandatory evaluation script for AI agents
├── demo.py                        # Local smoke test (no Docker, no HTTP)
├── pyproject.toml                 # Package configuration + dependencies
├── Dockerfile                     # Docker container specification
├── requirements.txt               # Installed dependencies
├── .env                           # Environment variables (IGNORED in GIT)
│
├── server/                        # HTTP API (FastAPI) Layer
│   ├── __init__.py                
│   ├── app.py                     # API endpoints (/reset, /step, /state, /health)
│   └── environment.py             # IncidentEnvironment class
│
└── envs/                          # Environment Engine
    ├── __init__.py
    ├── service_graph.py           # Topologies for microservices
    ├── incident_generator.py      # Generates simulated incidents
    ├── alert_generator.py         # Generates alerts based on incidents
    ├── grader.py                  # Evaluates agent actions
    ├── runbooks.py                # Simulated runbooks for resolution
    └── tasks.py                   # Task difficulty definitions
```

## 🛠 Requirements

* **Python 3.11+**
* **Docker** (Recommended for standalone evaluations)

### Environment Variables (`.env`)
You must configure the `.env` file in the root directory before running inference:
```env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=hf_your_huggingface_access_token
INCIDENT_SERVER_URL=http://localhost:7860
SEED=42
```

## 🐳 Running with Docker (Recommended)

To run the complete project using Docker, which encapsulates the FastAPI application and necessary modules:

1. **Build the container image**:
   ```bash
   docker build -t incidentmind .
   ```

2. **Run the container**:
   The server operates on port **7860**.
   ```bash
   docker run -d -p 7860:7860 --env-file .env --name incidentmind_server incidentmind
   ```

3. **Check the logs**:
   ```bash
   docker logs incidentmind_server
   ```

## 💻 Running Locally (Development Mode)

If you wish to test or develop without Docker:

1. **Install dependencies**:
   ```bash
   pip install -e .
   # OR
   pip install -r requirements.txt
   ```

2. **Run the local demo smoke test**:
   *(Bypasses HTTP/Docker, acts as a local module test)*
   ```bash
   python demo.py
   ```

3. **Start the local server**:
   ```bash
   uvicorn server.app:app --port 7860 --host 0.0.0.0
   ```

## 🤖 Running the AI Inference Agent

Once the server is running (either locally or via Docker on port 7860), you can evaluate the agent on the task sequences:

```bash
python inference.py
```
This script will:
* Check the `/health` endpoint of the Server.
* Play through `task1`, `task2`, and `task3` using the specified LLM.
* Evaluate actions and provide a Final Score average at the end.

## ⚠️ Important Development Guidelines & Rules

1. **Pydantic version**: The project strictly uses the Pydantic v2 API. Avoid using `class Config:` or `.dict()`. Always use `model_config = {}` and `.model_dump()`.
2. **Ports**: Port **7860** is mandatory across the stack (`Dockerfile`, `app.py`, internal tests).
3. **Environment Security**: NEVER commit your `.env` file or hardcode tokens (like `HF_TOKEN`) inside `inference.py`. Always fetch them using `os.environ.get()`.
4. **State Modifications**: The HTTP `GET /state` route must be purely read-only and never modify state, step count, or variables.
5. **JSON Responses**: `inference.py` ensures all returned values from the LLM are evaluated as strict JSON.
6. **Submissions**: Before final submission to the Hackathon, ensure your code is tagged correctly on main (`git tag v1.0.0`).