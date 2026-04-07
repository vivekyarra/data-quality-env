"""
FastAPI server exposing DataQualityEnv over HTTP.
"""

from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment import DataQualityEnv
from models import Action, Observation, ResetRequest, StateSnapshot

APP_NAME = "DataQualityEnv"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "OpenEnv environment for cross-domain operational data remediation on "
    "CRM, revenue-operations, and healthcare-billing tables."
)

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DataQualityEnv()


def _metadata_payload() -> Dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "mode": "simulation",
        "tasks": [t.model_dump() for t in env.list_tasks()],
        "endpoints": ["/health", "/metadata", "/schema", "/tasks", "/reset", "/step", "/state", "/mcp"],
    }


def _state_schema() -> Dict[str, Any]:
    return StateSnapshot.model_json_schema()


@app.get("/")
def root():
    return _metadata_payload()


@app.get("/metadata")
def metadata():
    return _metadata_payload()


@app.get("/health")
def health():
    return {"status": "healthy", "environment": APP_NAME}


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": _state_schema(),
    }


@app.post("/mcp")
def mcp_stub(payload: Optional[Dict[str, Any]] = Body(default=None)):
    request_id = payload.get("id") if isinstance(payload, dict) else None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": "MCP is not implemented for this environment.",
        },
    }


@app.get("/tasks")
def list_tasks():
    return [t.model_dump() for t in env.list_tasks()]


@app.post("/reset")
def reset(request: Optional[ResetRequest] = Body(default=None)):
    task_id = request.task_id if request is not None else "task1_easy"

    try:
        obs = env.reset(task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Action):
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/state")
def state():
    return env.state()


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
