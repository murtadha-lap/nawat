"""The control plane.

Everything the researcher can do from the shell is available here, over one
implementation (FR-1.11): storage, the registry, run submission and history with
live logs, and the inference session.

``/v1`` is a stable OpenAI-compatible address that outlives any individual
session (FR-4.5) — it forwards to whichever server is currently up, so a client
configured once keeps working across restarts, model changes and idle teardowns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Any, AsyncIterator, Iterator

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import metrics
from .cache import Cache, open_cache
from .config import Config
from .errors import NawatError
from .executor import Executor, RunQueue, process_alive
from .health import run_checks
from .keys import KINDS, Key
from .runs import RunSpec, RunState, RunStore, rebuild_from_disk
from .sessions import ServingBackend, SessionManager
from .units import human_bytes

IDLE_SWEEP_SECONDS = 15.0


# -- request bodies -----------------------------------------------------------


class SubmitBody(BaseModel):
    script: str = Field(description="Path to a .py or .ipynb inside the workspace")
    model: str | None = None
    datasets: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    params: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    run_id: str | None = None

    def to_spec(self) -> RunSpec:
        return RunSpec(
            script=self.script,
            model=Key.parse(self.model) if self.model else None,
            datasets=tuple(Key.parse(k) for k in self.datasets),
            inputs=tuple(Key.parse(k) for k in self.inputs),
            params=dict(self.params),
            notes=self.notes,
        )


class SessionBody(BaseModel):
    model: str
    idle_timeout: float | None = None
    extra_args: list[str] | None = None


class AdapterBody(BaseModel):
    key: str
    name: str | None = None


class ProposeBody(BaseModel):
    instruction: str
    script: str | None = None
    run_id: str | None = None


class ApplyBody(BaseModel):
    path: str
    content: str
    summary: str | None = None
    instruction: str | None = None
    backend: str | None = None


class EvaluateBody(BaseModel):
    data: str = Field(description="A dataset key or local path holding the eval JSONL")
    file: str | None = None
    base: str | None = None
    limit: int | None = None
    label: str | None = None
    per_sample: bool = False


class PublishBody(BaseModel):
    directory: str
    key: str
    keep_local: bool = False


class FreeBody(BaseModel):
    need: int = 0
    dry_run: bool = False


# -- application --------------------------------------------------------------


class Platform:
    """The objects the API is a thin skin over."""

    def __init__(
        self,
        config: Config | None = None,
        cache: Cache | None = None,
        backend: ServingBackend | None = None,
        *,
        start_queue: bool = True,
    ) -> None:
        self.cache = cache or open_cache(config)
        self.config = self.cache.config
        self.runs = RunStore(self.cache.db, self.config.state_dir / "runs")
        self.executor = Executor(self.cache, self.runs)
        self.queue = RunQueue(self.executor)
        self.sessions = SessionManager(self.cache, backend=backend, runs=self.runs)
        self._start_queue = start_queue

    def startup(self) -> None:
        self.cache.reconcile()
        rebuild_from_disk(self.runs)
        # Nothing survives a restart of the platform mid-run; say so rather than
        # leaving a record stuck in "running" forever.
        self.runs.reconcile(process_alive)
        if self._start_queue:
            self.queue.start()

    def shutdown(self) -> None:
        self.queue.stop()


def create_app(
    config: Config | None = None,
    cache: Cache | None = None,
    backend: ServingBackend | None = None,
    *,
    start_queue: bool = True,
    sweep_idle: bool = True,
) -> FastAPI:
    platform = Platform(config, cache, backend, start_queue=start_queue)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        platform.startup()
        sweeper = asyncio.create_task(_sweep_idle(platform)) if sweep_idle else None
        try:
            yield
        finally:
            if sweeper is not None:
                sweeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sweeper
            platform.shutdown()

    app = FastAPI(
        title="Nawāt",
        version="0.2.0",
        summary="Storage and orchestration core for a storage-constrained GPU host",
        lifespan=lifespan,
    )
    app.state.platform = platform

    # -- authentication ----------------------------------------------------

    def authorise(request: Request) -> None:
        """A token is required when one is configured (NFR-4.2)."""
        expected = platform.config.api_token
        if not expected:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-nawat-token", "")
        if supplied != expected:
            raise HTTPException(status_code=401, detail="A valid token is required. Set NAWAT_API_TOKEN and send it as a bearer token.")

    guard = [Depends(authorise)]

    @app.exception_handler(NawatError)
    async def nawat_error(request: Request, exc: NawatError) -> JSONResponse:
        """Errors keep their cause and their remedy across the wire (NFR-3.3)."""
        status = {2: 400, 3: 404, 4: 503, 5: 409, 6: 507, 7: 503, 8: 409}.get(exc.exit_code, 500)
        return JSONResponse(
            status_code=status,
            content={
                "error": type(exc).__name__,
                "cause": exc.cause,
                "remedy": exc.remedy,
                "detail": getattr(exc, "held", []),
            },
        )

    # -- platform ----------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        session = platform.sessions.current()
        active = platform.runs.active()
        status = platform.cache.status()
        return {
            "status": "ok",
            "cache": _status_json(platform.cache),
            "run": active[0].to_json() if active else None,
            "session": session.to_json() if session else None,
            "gpu": gpu_info(),
            "warnings": storage_warnings(status),
            "jupyter_url": os.environ.get("NAWAT_JUPYTER_URL") or None,
        }

    @app.get("/check", dependencies=guard)
    def check(create_bucket: bool = False) -> dict[str, Any]:
        checks = run_checks(platform.cache, create_bucket=create_bucket)
        return {
            "ready": all(c.ok for c in checks),
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail, "remedy": c.remedy} for c in checks],
        }

    @app.get("/config", dependencies=guard)
    def configuration() -> dict[str, Any]:
        return platform.config.redacted()

    # -- storage -----------------------------------------------------------

    @app.get("/cache", dependencies=guard)
    def cache_status() -> dict[str, Any]:
        return _status_json(platform.cache)

    @app.get("/cache/artifacts", dependencies=guard)
    def artifacts(kind: str | None = Query(default=None)) -> list[dict[str, Any]]:
        items = platform.cache.list()
        if kind:
            items = [a for a in items if a.key.kind == kind]
        return [_artifact_json(a) for a in items]

    @app.post("/cache/free", dependencies=guard)
    def free(body: FreeBody) -> dict[str, Any]:
        result = platform.cache.collect(body.need, dry_run=body.dry_run)
        return {
            "freed": result.freed,
            "freed_human": human_bytes(result.freed),
            "target": result.target,
            "satisfied": result.satisfied,
            "evicted": [str(k) for k in result.evicted],
            "kept": [{"key": str(k), "reason": r} for k, r in result.skipped],
        }

    @app.post("/cache/publish", dependencies=guard)
    def publish(body: PublishBody) -> dict[str, Any]:
        result = platform.cache.publish(body.directory, body.key, keep_local=body.keep_local)
        return {
            "key": str(result.key),
            "bytes": result.bytes,
            "files": result.verification.local_files,
            "local_removed": result.local_removed,
            "retained_reason": result.retained_reason,
        }

    @app.post("/cache/{key:path}/resolve", dependencies=guard)
    def resolve(key: str) -> dict[str, Any]:
        path = platform.cache.resolve(key, lease=False)
        status = platform.cache.get(key)
        return {"key": key, "path": str(path), "artifact": _artifact_json(status) if status else None}

    @app.post("/cache/{key:path}/keep", dependencies=guard)
    def keep(key: str) -> dict[str, Any]:
        return _artifact_json(platform.cache.pin(key))

    @app.delete("/cache/{key:path}/keep", dependencies=guard)
    def release(key: str) -> dict[str, Any]:
        return _artifact_json(platform.cache.unpin(key))

    @app.get("/cache/{key:path}/verify", dependencies=guard)
    def verify(key: str) -> dict[str, Any]:
        result = platform.cache.verify(key)
        return {
            "key": str(result.key),
            "ok": result.ok,
            "reason": result.reason(),
            "files": result.local_files,
            "bytes": result.local_bytes,
            "missing": list(result.missing),
            "size_mismatch": list(result.size_mismatch),
        }

    @app.delete("/cache/{key:path}", dependencies=guard)
    def evict(key: str, force: bool = False) -> dict[str, Any]:
        freed = platform.cache.evict(key, force=force)
        return {"key": key, "freed": freed, "freed_human": human_bytes(freed)}

    @app.get("/registry", dependencies=guard)
    def registry(kind: list[str] | None = Query(default=None)) -> list[dict[str, Any]]:
        cached = {str(a.key) for a in platform.cache.list()}
        keys = platform.cache.registry(kind or None)
        return [{"key": str(k), "cached": str(k) in cached} for k in sorted(keys, key=str)]

    @app.get("/leases", dependencies=guard)
    def leases() -> list[dict[str, Any]]:
        return [
            {
                "key": str(r.key),
                "holder": r.holder,
                "pid": r.pid,
                "acquired_at": r.acquired_at,
                "describe": r.describe(),
            }
            for r in platform.cache.leases.live()
        ]

    # -- workspace ---------------------------------------------------------

    @app.get("/scripts", dependencies=guard)
    def scripts() -> list[dict[str, Any]]:
        """Training scripts and notebooks available to submit (FR-3.4)."""
        root = platform.config.workspace_root
        found = []
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.suffix in (".py", ".ipynb") and not any(
                    part.startswith((".", "__")) for part in path.relative_to(root).parts
                ):
                    stat = path.stat()
                    found.append(
                        {
                            "path": str(path.relative_to(root)),
                            "kind": "notebook" if path.suffix == ".ipynb" else "script",
                            "bytes": stat.st_size,
                            "modified": stat.st_mtime,
                        }
                    )
        return found

    # -- runs --------------------------------------------------------------

    @app.post("/runs", status_code=201, dependencies=guard)
    def submit(body: SubmitBody) -> dict[str, Any]:
        record = platform.queue.submit(body.to_spec(), body.run_id)
        return record.to_json()

    @app.get("/runs", dependencies=guard)
    def list_runs(state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        parsed = RunState(state) if state else None
        return [record.to_json() for record in platform.runs.list(state=parsed, limit=limit)]

    @app.get("/runs/{run_id}", dependencies=guard)
    def get_run(run_id: str) -> dict[str, Any]:
        return platform.runs.get(run_id).to_json()

    @app.get("/runs/{run_id}/log", response_class=PlainTextResponse, dependencies=guard)
    def run_log(run_id: str, tail: int | None = None) -> str:
        platform.runs.get(run_id)
        return platform.runs.read_log(run_id, tail=tail)

    @app.get("/runs/{run_id}/log/stream", dependencies=guard)
    def stream_log(run_id: str) -> StreamingResponse:
        """Server-sent events, so a client sees a run as it happens (FR-2.6)."""
        platform.runs.get(run_id)

        def events() -> Iterator[bytes]:
            for chunk in platform.runs.follow_log(run_id):
                for line in chunk.splitlines():
                    yield f"data: {line}\n\n".encode()
            record = platform.runs.get(run_id)
            yield f"event: state\ndata: {json.dumps(record.to_json())}\n\n".encode()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/runs/{run_id}/cancel", dependencies=guard)
    def cancel(run_id: str) -> dict[str, Any]:
        return platform.executor.cancel(run_id).to_json()

    # -- agent -------------------------------------------------------------

    @app.get("/agent", dependencies=guard)
    def agent_status() -> dict[str, Any]:
        from .agent import build_backend

        try:
            backend = build_backend(workspace=platform.config.workspace_root)
            return {"configured": True, "backend": backend.name}
        except NawatError as exc:
            return {"configured": False, "backend": None, "remedy": exc.remedy}

    @app.post("/agent/propose", dependencies=guard)
    def agent_propose(body: ProposeBody) -> dict[str, Any]:
        from .agent import Agent

        agent = Agent(platform.cache, platform.runs)
        proposal = agent.propose(body.instruction, script=body.script, run_id=body.run_id)
        return proposal.to_json()

    @app.post("/agent/apply", dependencies=guard)
    def agent_apply(body: ApplyBody) -> dict[str, Any]:
        """The approval step: the reviewed content, applied and committed."""
        from .agent import Agent, Proposal

        agent = Agent(platform.cache, platform.runs)
        path = platform.config.workspace_root / body.path
        proposal = Proposal(
            path=body.path,
            old=path.read_text() if path.is_file() else "",
            new=body.content,
            summary=body.summary or "agent edit",
            backend=body.backend or "reviewed",
            instruction=body.instruction or "",
        )
        commit = agent.apply(proposal)
        return {"path": body.path, "commit": commit}

    @app.post("/runs/{run_id}/describe", dependencies=guard)
    def describe_run(run_id: str) -> dict[str, Any]:
        from .agent import Agent

        agent = Agent(platform.cache, platform.runs)
        return {"run_id": run_id, "description": agent.describe_run(run_id)}

    # -- evaluation --------------------------------------------------------

    @app.post("/runs/{run_id}/evaluate", dependencies=guard)
    def evaluate_run(run_id: str, body: EvaluateBody) -> dict[str, Any]:
        """Score the run's adapter against a held-out set (FR-4.7).

        Synchronous: evaluation sets are small and the caller wants the number.
        """
        from .evaluate import Evaluator

        evaluator = Evaluator(platform.cache, platform.runs, platform.sessions)
        result = evaluator.evaluate_run(
            run_id, body.data, base=body.base, file_name=body.file,
            limit=body.limit, label=body.label,
        )
        summary = result.to_json()
        if not body.per_sample:
            summary.pop("per_sample")
        return summary

    @app.get("/runs/{run_id}/evaluations", dependencies=guard)
    def list_evaluations(run_id: str) -> list[dict[str, Any]]:
        platform.runs.get(run_id)
        out = []
        for path in sorted(platform.runs.directory(run_id).glob("eval-*.json")):
            try:
                data = json.loads(path.read_text())
                data.pop("per_sample", None)
                out.append(data)
            except ValueError:
                continue
        return out

    # -- metrics -----------------------------------------------------------

    @app.get("/runs/{run_id}/metrics", dependencies=guard)
    def run_metrics(run_id: str) -> dict[str, Any]:
        """The full series, renderable long after the artifacts are gone (FR-7.6)."""
        platform.runs.get(run_id)
        points = metrics.read_points(platform.runs.metrics_path(run_id))
        return {
            "run_id": run_id,
            "points": len(points),
            "series": metrics.series(points),
            "events": metrics.events(points),
        }

    @app.get("/runs/{run_id}/metrics/stream", dependencies=guard)
    def stream_metrics(run_id: str) -> StreamingResponse:
        """Server-sent events: each point as it is written, live (FR-7.1)."""
        platform.runs.get(run_id)
        path = platform.runs.metrics_path(run_id)

        def running() -> bool:
            record = platform.runs.find(run_id)
            return record is not None and not record.state.terminal

        def event_stream() -> Iterator[bytes]:
            for point in metrics.follow(path, running):
                yield f"data: {json.dumps(point)}\n\n".encode()
            record = platform.runs.get(run_id)
            yield f"event: state\ndata: {json.dumps(record.to_json())}\n\n".encode()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/metrics/compare", dependencies=guard)
    def compare_metrics(
        run: list[str] = Query(default=[]),
        name: str = "loss",
    ) -> dict[str, list[dict[str, Any]]]:
        """One metric across several runs, on shared axes (FR-7.3)."""
        out: dict[str, list[dict[str, Any]]] = {}
        for run_id in run:
            platform.runs.get(run_id)
            points = metrics.read_points(platform.runs.metrics_path(run_id))
            out[run_id] = metrics.series(points).get(name, [])
        return out

    # -- sessions ----------------------------------------------------------

    @app.get("/sessions/current", dependencies=guard)
    def current_session() -> dict[str, Any] | None:
        session = platform.sessions.current()
        return session.to_json() if session else None

    @app.post("/sessions", dependencies=guard)
    def start_session(body: SessionBody) -> dict[str, Any]:
        session = platform.sessions.start(
            body.model, idle_timeout=body.idle_timeout, extra_args=body.extra_args
        )
        return session.to_json()

    @app.delete("/sessions", dependencies=guard)
    def stop_session() -> dict[str, Any]:
        session = platform.sessions.stop()
        return {"stopped": session.to_json() if session else None}

    @app.get("/sessions/log", response_class=PlainTextResponse, dependencies=guard)
    def session_log(tail: int = 200) -> str:
        return platform.sessions.log(tail=tail)

    @app.post("/sessions/adapters", dependencies=guard)
    def load_adapter(body: AdapterBody) -> dict[str, Any]:
        return platform.sessions.load_adapter(body.key, body.name).to_json()

    @app.delete("/sessions/adapters/{name}", dependencies=guard)
    def unload_adapter(name: str) -> dict[str, Any]:
        return platform.sessions.unload_adapter(name).to_json()

    # -- OpenAI-compatible passthrough -------------------------------------

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"], dependencies=guard)
    async def openai_proxy(path: str, request: Request) -> Any:
        """A stable address in front of an ephemeral server (FR-4.5)."""
        import httpx

        session = platform.sessions.require()
        platform.sessions.touch(session)
        url = f"{session.base_url}/v1/{path}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "authorization", "x-nawat-token")
        }
        body = await request.body()
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        upstream = client.build_request(request.method, url, headers=headers, content=body, params=request.query_params)
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            raise HTTPException(status_code=503, detail=f"The inference server did not answer: {exc}") from exc

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            },
            media_type=response.headers.get("content-type"),
        )

    # -- the index ---------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, Any]:
        """What this is and where its documentation lives.

        There is no bundled interface: the clients are the CLI, the Python
        library, and this schema.
        """
        from . import __version__

        return {
            "name": "nawat",
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
        }

    return app


_GPU_CACHE: dict[str, Any] = {"at": 0.0, "value": None}


def gpu_info() -> dict[str, Any] | None:
    """One reading from nvidia-smi, cached briefly. None when there is no GPU."""
    import subprocess

    now = time.time()
    if now - _GPU_CACHE["at"] < 2.0:
        return _GPU_CACHE["value"]
    value = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, used, total, util = [part.strip() for part in out.stdout.strip().splitlines()[0].split(",")]
            value = {
                "name": name,
                "memory_used": int(used) * 2**20,
                "memory_total": int(total) * 2**20,
                "utilization": int(util),
            }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        value = None
    _GPU_CACHE.update(at=now, value=value)
    return value


def storage_warnings(status) -> list[str]:
    """Storage-pressure conditions, surfaced prominently (FR-5.7)."""
    warnings: list[str] = []
    if status.ceiling and status.fraction >= 0.9:
        warnings.append(f"cache at {status.fraction * 100:.0f}% of its ceiling")
    if status.unreplicated_bytes:
        warnings.append(f"{human_bytes(status.unreplicated_bytes)} exists only on this disk")
    if status.disk_free < 5 * 10**9:
        warnings.append(f"only {human_bytes(status.disk_free)} free on the filesystem")
    return warnings


async def _sweep_idle(platform: Platform) -> None:
    """Release the GPU when nobody is using it (FR-4.4)."""
    while True:
        await asyncio.sleep(IDLE_SWEEP_SECONDS)
        try:
            await asyncio.to_thread(platform.sessions.reap_if_idle)
        except Exception:  # noqa: BLE001 - a sweep failure must not kill the loop
            continue


def _status_json(cache: Cache) -> dict[str, Any]:
    status = cache.status()
    return {
        "used": status.used,
        "used_human": human_bytes(status.used),
        "ceiling": status.ceiling,
        "ceiling_human": human_bytes(status.ceiling),
        "fraction": status.fraction,
        "artifacts": status.artifacts,
        "pinned_bytes": status.pinned_bytes,
        "leased_bytes": status.leased_bytes,
        "unreplicated_bytes": status.unreplicated_bytes,
        "disk_free": status.disk_free,
        "disk_total": status.disk_total,
    }


def _artifact_json(artifact) -> dict[str, Any]:
    return {
        "key": str(artifact.key),
        "bytes": artifact.bytes,
        "bytes_human": human_bytes(artifact.bytes),
        "files": artifact.files,
        "last_used": artifact.last_used,
        "pinned": artifact.pinned,
        "leased": artifact.leased,
        "replicated": artifact.replicated,
        "holders": [r.describe() for r in artifact.leases],
    }


def serve(config: Config | None = None) -> None:
    """Run the control plane. Bound to the configured host only (NFR-4.3)."""
    import uvicorn

    config = config or Config.from_env()
    uvicorn.run(create_app(config), host=config.api_host, port=config.api_port, log_level="info")
