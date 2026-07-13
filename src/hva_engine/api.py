from __future__ import annotations

import asyncio
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from hva_engine.blind_eval import BlindSample, build_blind_evaluation_store_from_env
from hva_engine.danmaku import dispatch_danmaku
from hva_engine.engine import EngineError, build_default_engine
from hva_engine.models import (
    BlindRatingRequest,
    CreateBlindTrialRequest,
    CreateMatchRequest,
    DanmakuRequest,
    GameEvent,
    MatchView,
    PublishStimulusRequest,
    SubmitActionRequest,
)

app = FastAPI(
    title="Human vs Agent Engine",
    version="0.1.0",
    description="Evaluation-first runtime for human-vs-agent strategy MODs.",
)
origins = os.getenv("HVA_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = build_default_engine()
blind_evaluations = build_blind_evaluation_store_from_env()
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class MatchSockets:
    def __init__(self) -> None:
        self.connections: dict[str, set[WebSocket]] = {}

    async def add(self, match_id: str, socket: WebSocket) -> None:
        await socket.accept()
        self.connections.setdefault(match_id, set()).add(socket)

    def remove(self, match_id: str, socket: WebSocket) -> None:
        self.connections.get(match_id, set()).discard(socket)

    async def broadcast(self, match_id: str, view: MatchView) -> None:
        for socket in list(self.connections.get(match_id, set())):
            try:
                await socket.send_json(view.model_dump(mode="json"))
            except RuntimeError:
                self.remove(match_id, socket)


sockets = MatchSockets()
mutation_lock = asyncio.Lock()


def _bad_request(exc: EngineError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _require_debug_token(token: str | None) -> None:
    configured = os.environ.get("HVA_DEBUG_TOKEN")
    if not configured or token is None or not secrets.compare_digest(token, configured):
        raise HTTPException(status_code=403, detail="Admin debug access is disabled or denied")


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/blind-eval", include_in_schema=False)
async def blind_evaluation_page() -> FileResponse:
    return FileResponse(static_dir / "blind_eval.html")


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "mods": len(engine.mods),
        "matches": len(engine.matches),
        "fact_store": engine.fact_store.name,
        "memory_store": engine.memory_store.backend_name,
        "world_store": engine.world_store.backend_name,
        "agent_runtime": engine.agent_runtime,
        "llm_mods": sorted(engine.llm_mod_ids),
        "character_cards": len(engine.character_cards.cards),
        "blind_evaluation_store": blind_evaluations.backend_name,
    }


@app.get("/api/mods")
async def list_mods() -> list[dict[str, object]]:
    return [mod.manifest() for mod in engine.mods.values()]


@app.get("/api/character-cards")
async def list_character_cards() -> list[dict[str, object]]:
    return engine.character_cards.catalog()


@app.post("/api/matches", response_model=MatchView, status_code=201)
async def create_match(request: CreateMatchRequest) -> MatchView:
    try:
        async with mutation_lock:
            return await run_in_threadpool(
                engine.create_match,
                request.mod_id,
                request.human_name,
                request.seed,
                request.mode,
                agent_tuning=request.agent_tuning,
                agent_characters=request.agent_characters,
                human_memory_id=request.human_memory_id,
                agent_memory_owner_ids=request.agent_memory_owner_ids,
                world_id=request.world_id,
                resume_world=request.resume_world,
            )
    except EngineError as exc:
        raise _bad_request(exc) from exc


@app.get("/api/worlds/{world_id}")
async def world_metadata(world_id: str) -> dict[str, object]:
    try:
        return engine.world_metadata(world_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/matches/{match_id}", response_model=MatchView)
async def get_match(match_id: str) -> MatchView:
    try:
        return engine.view(match_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/matches/{match_id}/actions", response_model=MatchView)
async def submit_action(match_id: str, request: SubmitActionRequest) -> MatchView:
    try:
        async with mutation_lock:
            view = await run_in_threadpool(
                engine.submit, match_id, request.actor_id, request.action
            )
        await sockets.broadcast(match_id, view)
        return view
    except EngineError as exc:
        raise _bad_request(exc) from exc


@app.get("/api/matches/{match_id}/evaluation")
async def get_evaluation(match_id: str) -> dict[str, object]:
    try:
        return engine.evaluation(match_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/evaluations/summary")
async def evaluation_summary() -> dict[str, object]:
    return engine.evaluation_summary()


@app.post("/api/evaluations/blind-trials", status_code=201)
async def create_blind_trial(request: CreateBlindTrialRequest) -> dict[str, object]:
    try:
        return blind_evaluations.create_trial(
            request.study_id,
            BlindSample(
                request.sample_a.condition_id,
                tuple(request.sample_a.transcript),
                request.sample_a.metadata,
            ),
            BlindSample(
                request.sample_b.condition_id,
                tuple(request.sample_b.transcript),
                request.sample_b.metadata,
            ),
            request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/evaluations/blind-ratings", status_code=201)
async def submit_blind_rating(request: BlindRatingRequest) -> dict[str, object]:
    try:
        return blind_evaluations.submit_rating(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/evaluations/blind-trials/{trial_id}")
async def get_blind_trial(trial_id: str) -> dict[str, object]:
    trial = blind_evaluations.get_trial(trial_id)
    if trial is None:
        raise HTTPException(status_code=404, detail="Unknown blind trial")
    return trial


@app.get("/api/evaluations/blind-summary/{study_id}")
async def blind_summary(study_id: str) -> dict[str, object]:
    return blind_evaluations.summary(study_id)


@app.get("/api/matches/{match_id}/agents/{agent_id}/context-preview")
async def context_preview(
    match_id: str,
    agent_id: str,
    x_hva_debug_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_debug_token(x_hva_debug_token)
    try:
        return engine.context_preview(match_id, agent_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/debug/matches/{match_id}", response_model=MatchView)
async def debug_match(
    match_id: str,
    x_hva_debug_token: str | None = Header(default=None),
) -> MatchView:
    _require_debug_token(x_hva_debug_token)
    try:
        return engine.debug_view(match_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/debug/matches/{match_id}/stimuli", response_model=GameEvent)
async def publish_debug_stimulus(
    match_id: str,
    request: PublishStimulusRequest,
    x_hva_debug_token: str | None = Header(default=None),
) -> GameEvent:
    """Trusted development adapter; production sensory sources need their own auth."""

    _require_debug_token(x_hva_debug_token)
    try:
        async with mutation_lock:
            event = await run_in_threadpool(
                engine.publish_stimulus,
                match_id,
                **request.model_dump(exclude_none=True),
            )
        await sockets.broadcast(match_id, engine.view(match_id))
        return event
    except EngineError as exc:
        raise _bad_request(exc) from exc


@app.get("/api/matches/{match_id}/agents/{agent_id}/fact-graph")
async def public_fact_graph(match_id: str, agent_id: str) -> dict[str, object]:
    try:
        return engine.public_fact_graph(match_id, agent_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/live/danmaku", response_model=MatchView)
async def danmaku(request: DanmakuRequest) -> MatchView:
    try:
        async with mutation_lock:
            view = await run_in_threadpool(
                dispatch_danmaku,
                engine,
                request.match_id,
                request.message,
                request.user,
            )
        await sockets.broadcast(request.match_id, view)
        return view
    except EngineError as exc:
        raise _bad_request(exc) from exc


@app.websocket("/ws/matches/{match_id}")
async def match_stream(websocket: WebSocket, match_id: str) -> None:
    try:
        view = engine.view(match_id)
    except EngineError:
        await websocket.close(code=4404)
        return
    await sockets.add(match_id, websocket)
    await websocket.send_json(view.model_dump(mode="json"))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        sockets.remove(match_id, websocket)
