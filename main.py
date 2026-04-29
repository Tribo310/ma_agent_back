from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

import traceback

import model_store
from battle_runner import run_battle

app = FastAPI(title="MAgent Battle API")

# ---------------------------------------------------------------------------
# CORS
# Vercel preview deploys use *.vercel.app; local dev uses localhost:5173.
# allow_origin_regex handles the wildcard subdomain pattern.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/test-env")
async def test_env() -> dict:
    """Diagnose magent2 import and env creation."""
    try:
        from magent2.environments import battle_v4
        env = battle_v4.parallel_env(map_size=15, max_cycles=3)
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        n = len(env.agents)
        env.close()
        return {"status": "ok", "agents": n, "sample_agent": list(obs.keys())[0]}
    except Exception as exc:
        return {"status": "error", "detail": traceback.format_exc(), "error": str(exc)}


@app.get("/ready")
async def ready() -> dict[str, bool]:
    return {
        "ready": model_store.is_ready(),
        "red": model_store.red_ready(),
        "blue": model_store.blue_ready(),
    }


@app.post("/upload/{team}")
async def upload(team: str, file: UploadFile = File(...)) -> dict[str, str]:
    if team not in ("red", "blue"):
        raise HTTPException(status_code=400, detail="team must be 'red' or 'blue'")

    content = await file.read()
    try:
        model_store.validate_and_store(team, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"team": team, "status": "ready"}


# ---------------------------------------------------------------------------
# WebSocket battle endpoint
# ---------------------------------------------------------------------------

@app.websocket("/battle")
async def battle_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    if not model_store.is_ready():
        await websocket.send_json({"error": "Upload both models before starting a battle."})
        await websocket.close(code=1008)
        return

    red_model = model_store.get_model("red")
    blue_model = model_store.get_model("blue")

    try:
        await run_battle(websocket, red_model, blue_model)  # type: ignore[arg-type]
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        detail = traceback.format_exc()
        print("BATTLE ERROR:\n", detail)          # visible in Railway logs
        try:
            await websocket.send_json({"error": str(exc), "detail": detail})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
