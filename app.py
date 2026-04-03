"""
app.py — FastAPI server exposing DataQualityEnv over HTTP (OpenEnv spec).

Endpoints
---------
GET  /              → environment info
GET  /health        → liveness probe
GET  /tasks         → list available tasks
POST /reset         → start a new episode   body: {"task_id": "task1_easy"}
POST /step          → apply one action      body: Action JSON
GET  /state         → current episode state

Run locally:
    uvicorn app:app --reload --port 7860

HF Spaces listens on port 7860 by default.
"""

from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import Action, ResetRequest
from environment import DataQualityEnv

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DataQualityEnv",
    description=(
        "An OpenEnv-compliant reinforcement learning environment for training AI agents "
        "to perform data quality / cleaning tasks on real-world tabular datasets."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful, session-based for simplicity)
env = DataQualityEnv()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":        "DataQualityEnv",
        "version":     "0.1.0",
        "description": "OpenEnv environment for data-quality / cleaning tasks",
        "tasks":       [t.task_id for t in env.list_tasks()],
        "endpoints":   ["/health", "/tasks", "/reset", "/step", "/state"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "environment": "DataQualityEnv"}


@app.get("/tasks")
def list_tasks():
    """Return metadata for all available tasks."""
    return [t.model_dump() for t in env.list_tasks()]


@app.post("/reset")
def reset(request: Optional[ResetRequest] = Body(default=None)):
    """
    Start (or restart) an episode.

    Body: `{"task_id": "task1_easy"}` — defaults to task1_easy.
    Returns the initial Observation.
    """
    task_id = request.task_id if request is not None else "task1_easy"

    try:
        obs = env.reset(task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Action):
    """
    Apply one action to the environment.

    Body: `{"operation": "...", "column": "...", "params": {...}}`
    Returns StepResult (observation, reward, done, info).
    """
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/state")
def state():
    """Return the current episode state (lightweight, no full table)."""
    return env.state()
