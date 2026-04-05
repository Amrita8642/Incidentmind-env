# IncidentMind — Ritu's Reference Card
# Steps 8 & 9: Folder Structure + Common Mistakes

# ═══════════════════════════════════════════════════════════════════
# STEP 8 — FOLDER STRUCTURE (Ritu's files only, marked with ★)
# ═══════════════════════════════════════════════════════════════════

incidentmind/
│
├── __init__.py                    ★ Ritu — package exports
├── models.py                      ★ Ritu — ALL Pydantic models (write first)
├── client.py                      ★ Ritu — IncidentEnvClient (async + sync)
├── inference.py                   ★ Ritu — mandatory evaluation script (judges run this)
├── demo.py                        ★ Ritu — local smoke test (no Docker, no HTTP)
├── pyproject.toml                 ★ Ritu — package config + dependencies
│
├── server/
│   ├── __init__.py                ★ Ritu — empty, marks server as a package
│   ├── app.py                     ★ Ritu — FastAPI endpoints
│   ├── environment.py             ★ Ritu — IncidentEnvironment class
│   └── requirements.txt           ★ Ritu writes pyproject.toml;
│                                           Amrita generates requirements.txt from it
│
├── envs/                          ← Sneha owns ALL of these
│   ├── __init__.py
│   ├── service_graph.py
│   ├── incident_generator.py
│   ├── alert_generator.py
│   ├── grader.py
│   ├── runbooks.py
│   └── tasks.py
│
├── Dockerfile                     ← Amrita owns
├── openenv.yaml                   ← Amrita owns
├── README.md                      ← Amrita owns
└── .github/
    └── workflows/
        └── ci.yml                 ← Amrita owns

# NOTE: server/__init__.py is a blank file. Create it with:
#   touch server/__init__.py
# Without it, "from server.environment import IncidentEnvironment"
# will throw ModuleNotFoundError.


# ═══════════════════════════════════════════════════════════════════
# STEP 9 — COMMON MISTAKES THAT WILL BREAK THE PROJECT
# ═══════════════════════════════════════════════════════════════════

# ── MISTAKE 1: Missing server/__init__.py ─────────────────────────
# SYMPTOM: ModuleNotFoundError: No module named 'server'
# FIX: touch server/__init__.py
# WHY: Python needs __init__.py to treat a directory as a package.
#      server/app.py and server/environment.py are NOT importable
#      until this file exists.

# ── MISTAKE 2: Editing models.py after Amrita starts openenv.yaml ─
# SYMPTOM: openenv validate fails — field names in YAML don't match models.py
# FIX: Freeze models.py on Day 1. Any change requires telling Amrita immediately.
# WHY: The 12 observation fields and 7 action types must match exactly in both
#      models.py AND openenv.yaml. A rename in one without updating the other
#      will fail the automated judging check.

# ── MISTAKE 3: Hardcoding ANY value in inference.py ──────────────
# SYMPTOM: Disqualification from the hackathon
# FIX: Every URL, model name, and token MUST come from os.environ.get()
# NEVER write:
#   base_url = "https://api-inference.huggingface.co/v1"  # ← WRONG
# ALWAYS write:
#   base_url = os.environ.get("API_BASE_URL", "")

# ── MISTAKE 4: Committing .env to git ────────────────────────────
# SYMPTOM: HF_TOKEN exposed publicly → security incident → token rotation required
# FIX: .env is in .gitignore (Amrita added it). Never use git add -f .env
# Rule: if you can see HF_TOKEN in a git diff, STOP and rotate the token.

# ── MISTAKE 5: Starting server/environment.py before Sneha merges ─
# SYMPTOM: ImportError: cannot import name 'ServiceGraph' from 'envs.service_graph'
# FIX: Write models.py and pyproject.toml first (no envs imports).
#      Only write environment.py AFTER git merge origin/feat/sneha-env-engine.
# Coordination tip: Ask Sneha to push a stub __init__.py for each envs module
#      so your imports don't break while she's still coding.

# ── MISTAKE 6: Pydantic v1 vs v2 API confusion ───────────────────
# SYMPTOM: AttributeError: 'Observation' object has no attribute 'dict'
# FIX: You must use Pydantic v2 API (pydantic>=2.0.0 in pyproject.toml)
#      v2 uses .model_dump() not .dict()
#      v2 uses .model_validate() not .parse_obj()
#      v2 uses model_config = {} not class Config:
# If you see .dict() anywhere, it's Pydantic v1 code — update it.

# ── MISTAKE 7: Wrong port in uvicorn startup ──────────────────────
# SYMPTOM: Amrita's CI fails — curl /health returns connection refused
# FIX: Port MUST be 7860 everywhere — server/app.py, Dockerfile CMD, openenv.yaml
# Check: grep -r "7860" . — should appear in all three files.

# ── MISTAKE 8: state() modifying environment state ───────────────
# SYMPTOM: Calling GET /state changes step_count or resets something
# FIX: state() must be a pure read — no attribute assignments inside it.
#      It only reads self._episode_id, self._step_count, etc.

# ── MISTAKE 9: Returning wrong types from Sneha's modules ─────────
# SYMPTOM: AttributeError in _to_alert_model() — sneha_alert has no attribute 'id'
# FIX: Coordinate field names with Sneha on Day 4 checkpoint.
#      If Sneha uses 'alert_id' instead of 'id', update _to_alert_model() in
#      environment.py — NOT models.py (models.py is frozen).

# ── MISTAKE 10: Missing 'requests' in pyproject.toml ─────────────
# SYMPTOM: inference.py crashes with ModuleNotFoundError: No module named 'requests'
# FIX: Add "requests>=2.28.0" to pyproject.toml dependencies.
#      It's usually pre-installed but must be in requirements.txt for Docker.

# ── MISTAKE 11: Calling step() after done=True ───────────────────
# SYMPTOM: environment.step() returns reward=0 with info["error"] on every call
# FIX: inference.py must check `if result.done: break` after every step.
#      The while loop must terminate when done=True.

# ── MISTAKE 12: LLM returns markdown instead of raw JSON ─────────
# SYMPTOM: parse_llm_response returns None → RESOLVE fallback triggered every step
# FIX: The system prompt says "respond with ONLY valid JSON".
#      If the model still adds markdown, parse_llm_response strips ``` fences.
#      If it still fails, add more examples to the system prompt.

# ── MISTAKE 13: Not creating the conversation history list ────────
# SYMPTOM: LLM forgets context, makes random actions every step
# FIX: inference.py maintains a `conversation` list and appends every
#      user prompt + assistant response to it. Passed to llm_client as `messages`.

# ── MISTAKE 14: CORS not enabled in app.py ───────────────────────
# SYMPTOM: HF Space web interface shows "CORS error" in browser console
# FIX: CORSMiddleware with allow_origins=["*"] is already in app.py.
#      Never remove it.

# ── MISTAKE 15: Not tagging the final commit ─────────────────────
# SYMPTOM: Hackathon submission rejected — no tag found
# FIX: After Amrita's final merge to main:
#   git tag v1.0.0
#   git push origin v1.0.0
