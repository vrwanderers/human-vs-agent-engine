from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hva_engine.danmaku import dispatch_danmaku
from hva_engine.engine import EngineError, build_default_engine
from hva_engine.models import CreateMatchRequest, DanmakuRequest, MatchView, SubmitActionRequest

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


def _bad_request(exc: EngineError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "mods": len(engine.mods),
        "matches": len(engine.matches),
        "fact_store": engine.fact_store.name,
        "agent_runtime": engine.agent_runtime,
        "llm_mods": sorted(engine.llm_mod_ids),
        "character_cards": len(engine.character_cards.cards),
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
        return engine.create_match(
            request.mod_id,
            request.human_name,
            request.seed,
            request.mode,
            agent_tuning=request.agent_tuning,
            agent_characters=request.agent_characters,
        )
    except EngineError as exc:
        raise _bad_request(exc) from exc


@app.get("/api/matches/{match_id}", response_model=MatchView)
async def get_match(match_id: str) -> MatchView:
    try:
        return engine.view(match_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/matches/{match_id}/actions", response_model=MatchView)
async def submit_action(match_id: str, request: SubmitActionRequest) -> MatchView:
    try:
        view = engine.submit(match_id, request.actor_id, request.action)
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


@app.get("/api/matches/{match_id}/agents/{agent_id}/context-preview")
async def context_preview(match_id: str, agent_id: str) -> dict[str, object]:
    try:
        return engine.context_preview(match_id, agent_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/matches/{match_id}/agents/{agent_id}/fact-graph")
async def public_fact_graph(match_id: str, agent_id: str) -> dict[str, object]:
    try:
        return engine.public_fact_graph(match_id, agent_id)
    except EngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/live/danmaku", response_model=MatchView)
async def danmaku(request: DanmakuRequest) -> MatchView:
    try:
        view = dispatch_danmaku(engine, request.match_id, request.message, request.user)
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
