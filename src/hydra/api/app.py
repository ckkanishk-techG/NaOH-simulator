"""FastAPI application: REST + WebSockets + static frontend + offline docs, served entirely from local files.

Laptop mode: loopback only, no auth. Server mode: local accounts (hashed passwords, roles), signed tokens, audit log,
job queue with CPU/RAM limits, backups. No physics lives in this layer."""

from __future__ import annotations

import asyncio
import html
import io
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__, backup, export, plots, report
from ..auth import AuthError, AuthStore, User, allowed
from ..config import Settings
from ..core.params import ParamSet
from ..core.safety import DISCLAIMER, analyze
from ..core.scenario import Scenario
from ..db import Database
from ..eln.store import ELN
from ..jobs import HANDLERS, JobQueue
from ..runs import RunRepository
from ..thermo import propdb
from . import handlers
from .schemas import ElnCreate, LoginRequest, ScaleupRequest, SimulateRequest, UserCreate

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
MAX_BODY = 50 * 1024 * 1024  # magic: 50 MB request cap
LOCAL_ADMIN = User("local", "admin", 0)


class Health(BaseModel):
    status: str
    version: str
    mode: str
    telemetry: bool
    offline_enforced: bool
    auth_required: bool
    property_db: str


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(settings.db_url)
        self.runs = RunRepository(self.db, settings.data_dir)
        self.eln = ELN(self.db, settings.data_dir)
        self.jobs = JobQueue(self.db, workers=max(settings.max_job_cpus, 2), max_cpus=settings.max_job_cpus, max_ram_mb=settings.max_job_ram_mb,
                             isolate=settings.mode == "server")
        self.auth: AuthStore | None = AuthStore(self.db, settings.data_dir / "secret.key") if settings.mode == "server" else None
        handlers.bind(self)


def render_markdown(md: str) -> str:
    """Minimal offline Markdown renderer (headings, lists, code fences, tables, bold/italic/code, links)."""
    out: list[str] = []
    in_code, in_list = False, False
    table: list[str] = []

    def inline(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", t)

    def flush_table() -> None:
        if table:
            rows = [r for r in table if not re.match(r"^\|?\s*[-:| ]+\|?$", r)]
            out.append("<table>" + "".join("<tr>" + "".join(f"<td>{inline(c.strip())}</td>" for c in r.strip("|").split("|")) + "</tr>" for r in rows) + "</table>")
            table.clear()

    for line in md.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            out.append("<pre>" if in_code else "</pre>")
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if in_list and not line.startswith(("- ", "* ")):
            out.append("</ul>")
            in_list = False
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
        elif line.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def docs_page(name: str) -> str:
    if name == "ASSUMPTIONS":  # generated live from the property database (units, ranges, status, sources)
        rows = "".join(f"| {r['name']} | {r['value']:.4g} | {r['unit']} | {r['min']} – {r['max']} | {r['status']} | {r['source']} |\n" for r in propdb.table())
        return ("# Assumptions table\n\nEvery constant: value, unit, plausible range, status (**prior** = uncalibrated, **unverified** = typed from memory, check it), "
                "source placeholder.\n\n| name | value | unit | range | status | source |\n|---|---|---|---|---|---|\n" + rows)
    p = DOCS_DIR / f"{name}.md"
    if not p.exists() or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise HTTPException(404, "no such page")
    return p.read_text()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state = AppState(settings)
    app = FastAPI(title="HYDRA", version=__version__, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.hydra = state

    @app.middleware("http")
    async def security(request: Request, call_next: Callable[..., Any]) -> Response:
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_BODY:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        resp: Response = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"
        if request.url.path.startswith("/api"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---------------------------------------------------------------- auth dependencies
    def user_of(request: Request) -> User:
        if state.auth is None:
            return LOCAL_ADMIN
        h = request.headers.get("authorization", "")
        if not h.lower().startswith("bearer "):
            raise HTTPException(401, "authentication required")
        try:
            return state.auth.verify_token(h[7:])
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc

    def need(role: str) -> Callable[..., User]:
        def dep(u: User = Depends(user_of)) -> User:
            if not allowed(u.role, role):
                raise HTTPException(403, f"requires role {role}")
            return u

        return dep

    def audit(u: User, action: str, detail: str = "") -> None:
        if state.auth is not None:
            state.auth.log(u.username, action, detail)

    # ---------------------------------------------------------------- basics
    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version=__version__, mode=settings.mode, telemetry=False, offline_enforced=settings.enforce_offline,
                      auth_required=state.auth is not None, property_db=propdb.db_hash())

    @app.get("/api/disclaimer")
    def disclaimer() -> JSONResponse:
        return JSONResponse({"text": DISCLAIMER})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/properties")
    def properties(category: str | None = None, _: User = Depends(need("viewer"))) -> list[dict[str, Any]]:
        return propdb.table(category)

    @app.get("/docs", response_class=HTMLResponse)
    def docs_index() -> str:
        names = sorted(p.stem for p in DOCS_DIR.glob("*.md")) + ["ASSUMPTIONS"]
        body = "<h1>HYDRA documentation (offline)</h1><ul>" + "".join(f"<li><a href='/docs/{n}'>{n}</a></li>" for n in names) + "</ul>"
        return f"<html><head><meta charset='utf-8'><title>HYDRA docs</title></head><body style='font:14px sans-serif;max-width:900px;margin:2em auto'>{body}</body></html>"

    @app.get("/docs/{name}", response_class=HTMLResponse)
    def docs_view(name: str) -> str:
        return ("<html><head><meta charset='utf-8'><title>HYDRA docs</title><style>body{font:14px/1.5 sans-serif;max-width:960px;margin:2em auto;padding:0 1em}"
                "td,th{border:1px solid #bbb;padding:2px 8px;font-size:12px}pre{background:#f4f4f4;padding:8px;overflow:auto}</style></head><body>"
                f"<p><a href='/docs'>&larr; docs</a></p>{render_markdown(docs_page(name))}</body></html>")

    # ---------------------------------------------------------------- auth endpoints
    @app.post("/api/auth/login")
    def login(req: LoginRequest) -> dict[str, str]:
        if state.auth is None:
            return {"token": "", "role": "admin", "mode": "laptop (no authentication)"}
        try:
            tok = state.auth.login(req.username, req.password)
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc
        return {"token": tok, "role": state.auth.verify_token(tok).role}

    @app.get("/api/auth/me")
    def me(u: User = Depends(user_of)) -> dict[str, str]:
        return {"username": u.username, "role": u.role}

    @app.post("/api/auth/bootstrap")
    def bootstrap(req: UserCreate) -> dict[str, str]:
        """Create the FIRST admin (only when no users exist)."""
        if state.auth is None:
            raise HTTPException(400, "authentication is disabled in laptop mode")
        if state.auth.n_users() > 0:
            raise HTTPException(403, "already bootstrapped")
        try:
            u = state.auth.create_user(req.username, req.password, "admin")
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"username": u.username, "role": u.role}

    @app.post("/api/auth/users")
    def create_user(req: UserCreate, u: User = Depends(need("admin"))) -> dict[str, str]:
        assert state.auth is not None
        try:
            nu = state.auth.create_user(req.username, req.password, req.role)
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from exc
        audit(u, "create_user", req.username)
        return {"username": nu.username, "role": nu.role}

    @app.get("/api/auth/users")
    def users(_: User = Depends(need("admin"))) -> list[dict[str, Any]]:
        return state.auth.list_users() if state.auth else []

    @app.get("/api/auth/audit")
    def audit_log(_: User = Depends(need("admin"))) -> list[dict[str, Any]]:
        return state.auth.audit() if state.auth else []

    # ---------------------------------------------------------------- safety / scenarios
    @app.post("/api/scenarios/validate")
    def validate_scenario(sc: Scenario, _: User = Depends(need("viewer"))) -> dict[str, Any]:
        return {"valid": True, "scenario": sc.model_dump()}

    @app.post("/api/safety/screen")
    def safety(sc: Scenario, _: User = Depends(need("viewer"))) -> dict[str, Any]:
        return handlers.safety_screen(sc)

    # ---------------------------------------------------------------- runs
    @app.post("/api/runs")
    def run_sim(req: SimulateRequest, u: User = Depends(need("researcher"))) -> dict[str, Any]:
        pre = handlers.safety_screen(req.scenario, req.overrides)
        from ..core.l1 import SolverSettings

        rec, res = state.runs.run(req.scenario, req.overrides, SolverSettings(rtol=req.rtol, dt_out=req.dt_out), u.username)
        saf = analyze(res, ParamSet(req.overrides))
        audit(u, "run", rec.run_id)
        return {"run_id": rec.run_id, "summary": res.summary, "ledger": res.ledger, "prescreen": pre,
                "safety": {"level": saf.level, "flags": [f.__dict__ for f in saf.flags], "forecasts": saf.forecasts},
                "range_violations": res.diagnostics.get("range_violations", [])}

    @app.get("/api/runs")
    def list_runs(_: User = Depends(need("viewer"))) -> list[dict[str, Any]]:
        return state.runs.list()

    def _run(run_id: str) -> dict[str, np.ndarray]:
        try:
            return state.runs.series(run_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, "no such run") from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, _: User = Depends(need("viewer"))) -> dict[str, Any]:
        try:
            return {"record": json.loads(state.runs.record(run_id).model_dump_json()), **state.runs.summary(run_id)}
        except KeyError as exc:
            raise HTTPException(404, "no such run") from exc

    @app.get("/api/runs/{run_id}/data")
    def run_data(run_id: str, fmt: str = "json", keys: str | None = None, _: User = Depends(need("viewer"))) -> Response:
        s = _run(run_id)
        ks = keys.split(",") if keys else None
        if fmt == "csv":
            return Response(export.to_csv(s, ks), media_type="text/csv")
        if fmt == "json":
            return JSONResponse({k: v.tolist() for k, v in s.items() if (ks is None or k in ks or k == "t")})
        if fmt in ("npz", "h5"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / f"{run_id}.{fmt}"
                if fmt == "npz":
                    export.to_npz(s, p)
                else:
                    export.to_hdf5(s, p, {"record": json.loads(state.runs.record(run_id).model_dump_json())})
                return Response(p.read_bytes(), media_type="application/octet-stream")
        raise HTTPException(400, "fmt must be json, csv, npz or h5")

    class _View:
        """Adapter exposing stored arrays with the attributes the plot functions expect."""

        def __init__(self, run_id: str) -> None:
            self.series = _run(run_id)
            self.t = self.series["t"]
            self.summary = state.runs.summary(run_id)["summary"]
            self.ledger = state.runs.summary(run_id)["ledger"]
            self.diagnostics = state.runs.summary(run_id)["diag"]
            self.model = None

        def __getitem__(self, k: str) -> np.ndarray:
            return self.series[k]

    @app.get("/api/runs/{run_id}/plot/{name}")
    def run_plot(run_id: str, name: str, fmt: str = "png", dpi: int = 300, theme: str = "light", _: User = Depends(need("viewer"))) -> Response:
        fn: Any = plots.PLOTS.get(name)
        if fn is None or name == "particles":
            raise HTTPException(404, f"plots: {sorted(set(plots.PLOTS) - {'particles'})}")
        if fmt not in ("png", "svg", "pdf") or not (30 <= dpi <= 1200):  # magic: sane dpi window
            raise HTTPException(400, "bad fmt/dpi")
        data = plots.render(fn(_View(run_id), theme=theme), fmt, dpi)
        return Response(data, media_type={"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[fmt])

    @app.get("/api/runs/{run_id}/report")
    def run_report(run_id: str, fmt: str = "html", _: User = Depends(need("viewer"))) -> Response:
        view = _View(run_id)
        rec = state.runs.record(run_id)
        doc = report.build_html(f"HYDRA run {run_id}", rec, view)
        if fmt == "pdf":
            return Response(report.build_pdf(f"HYDRA run {run_id}", doc, view), media_type="application/pdf")
        return HTMLResponse(doc)

    @app.post("/api/runs/{run_id}/rerun")
    def rerun(run_id: str, _: User = Depends(need("researcher"))) -> dict[str, Any]:
        try:
            same, res = state.runs.rerun(run_id)
        except KeyError as exc:
            raise HTTPException(404, "no such run") from exc
        return {"bit_identical": same, "h2_total_mol": res.summary["h2_total_mol"]}

    @app.post("/api/compare")
    def compare(run_ids: list[str], keys: list[str] | None = None, _: User = Depends(need("viewer"))) -> dict[str, Any]:
        keys = keys or ["h2_stp_L", "T", "P"]
        out: dict[str, Any] = {}
        for rid in run_ids:
            s = _run(rid)
            out[rid] = {"summary": state.runs.summary(rid)["summary"], **{k: s[k].tolist() for k in keys if k in s}, "t": s["t"].tolist()}
        return out

    # ---------------------------------------------------------------- jobs
    @app.get("/api/jobs/kinds")
    def job_kinds(_: User = Depends(need("viewer"))) -> list[str]:
        return sorted(HANDLERS)

    @app.post("/api/jobs/{kind}")
    def submit(kind: str, params: dict[str, Any], u: User = Depends(need("researcher"))) -> dict[str, str]:
        if kind not in HANDLERS:
            raise HTTPException(404, f"unknown job kind; available: {sorted(HANDLERS)}")
        jid = state.jobs.submit(kind, params, u.username)
        audit(u, "submit", f"{kind}:{jid}")
        return {"job_id": jid}

    @app.get("/api/jobs")
    def list_jobs(u: User = Depends(need("viewer"))) -> list[dict[str, Any]]:
        return state.jobs.list(None if allowed(u.role, "admin") else u.username)

    @app.get("/api/jobs/id/{jid}")
    def get_job(jid: str, u: User = Depends(need("viewer"))) -> dict[str, Any]:
        j = state.jobs.get(jid)
        if j is None or (j["owner"] != u.username and not allowed(u.role, "admin") and state.auth is not None):
            raise HTTPException(404, "no such job")
        return j

    @app.delete("/api/jobs/id/{jid}")
    def cancel(jid: str, u: User = Depends(need("researcher"))) -> dict[str, bool]:
        return {"cancelled": state.jobs.cancel(jid)}

    @app.websocket("/ws/jobs/{jid}")
    async def ws_job(ws: WebSocket, jid: str) -> None:
        await ws.accept()
        try:
            if state.auth is not None:
                state.auth.verify_token(ws.query_params.get("token", ""))
            last = None
            while True:
                j = state.jobs.get(jid)
                if j is None:
                    await ws.send_json({"error": "no such job"})
                    break
                snap = {"status": j["status"], "progress": j["progress"], "message": j["message"]}
                if snap != last:
                    await ws.send_json(snap)
                    last = snap
                if j["status"] in ("done", "failed", "cancelled"):
                    await ws.send_json({"final": True, "result": j.get("result"), "error": j.get("error")})
                    break
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, AuthError):
            pass
        await ws.close()

    @app.websocket("/ws/twin")
    async def ws_twin(ws: WebSocket) -> None:
        """Live twin stream. First message: ``{"scenario": {...}, "hidden": {...}, "method": "ekf"|"enkf", "n_frames": N}`` (simulated
        sensors in this endpoint; serial/CSV sources are read by the local process, never remote clients)."""
        await ws.accept()
        try:
            if state.auth is not None:
                state.auth.verify_token(ws.query_params.get("token", ""))
            cfg = await ws.receive_json()
            from ..hardware import sources, twin

            sc = Scenario.model_validate(cfg["scenario"])
            src = sources.SimulatedSource(sc, cfg.get("hidden"), dt=float(cfg.get("dt", 5.0)), stack_current=float(cfg.get("stack_current", 0.0)))
            tw = twin.LiveTwin(sc, cfg=twin.TwinConfig(method=cfg.get("method", "ekf"), limit_T_C=float(cfg.get("limit_T_C", 70.0))))
            for i, f in enumerate(src):
                if i >= int(cfg.get("n_frames", 100)):
                    break
                cum = float(np.interp(f.t, src.truth["t"], src.truth["gen"]))
                st = tw.ingest(f, cum)
                msg: dict[str, Any] = {"frame": f.__dict__, "state": st}
                if i % 10 == 9:  # magic: forecast every 10 frames
                    fc = tw.forecast(300.0, 15.0, 40)
                    msg["forecast"] = {"t": fc.t.tolist(), "T_lo": fc.T_lo.tolist(), "T_med": fc.T_med.tolist(), "T_hi": fc.T_hi.tolist(),
                                       "p_exceed_T": fc.p_exceed_T}
                    msg["warnings"] = [w.__dict__ for w in tw.warnings(300.0)]
                await ws.send_json(msg)
        except (WebSocketDisconnect, AuthError):
            return
        await ws.close()

    # ---------------------------------------------------------------- scale-up (sync, cheap)
    @app.post("/api/scaleup")
    def scaleup(req: ScaleupRequest, _: User = Depends(need("viewer"))) -> dict[str, Any]:
        out = handlers.scaleup_compute(req)
        return json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

    # ---------------------------------------------------------------- ELN
    @app.post("/api/eln/experiments")
    def eln_create(req: ElnCreate, u: User = Depends(need("researcher"))) -> dict[str, str]:
        try:
            eid = state.eln.create_experiment(req.title, req.operator or u.username, req.setup, req.quantities, req.conditions, req.notes, req.batch, req.split)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        audit(u, "eln_create", eid)
        return {"id": eid}

    @app.get("/api/eln/experiments")
    def eln_list(_: User = Depends(need("viewer"))) -> list[dict[str, Any]]:
        return state.eln.list_experiments()

    @app.get("/api/eln/experiments/{eid}")
    def eln_get(eid: str, _: User = Depends(need("viewer"))) -> dict[str, Any]:
        try:
            return state.eln.get(eid)
        except KeyError as exc:
            raise HTTPException(404, "no such experiment") from exc

    @app.put("/api/eln/experiments/{eid}/attachments/{filename}")
    async def eln_attach(eid: str, filename: str, request: Request, kind: str = "other", u: User = Depends(need("researcher"))) -> dict[str, Any]:
        data = await request.body()
        try:
            return state.eln.add_attachment(eid, filename, data, kind)
        except KeyError as exc:
            raise HTTPException(404, "no such experiment") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/eln/attachments/{aid}")
    def eln_download(aid: str, _: User = Depends(need("viewer"))) -> Response:
        try:
            name, data = state.eln.read_attachment(aid)
        except KeyError as exc:
            raise HTTPException(404, "no such attachment") from exc
        return Response(data, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/api/eln/experiments/{eid}/link-run/{run_id}")
    def eln_link(eid: str, run_id: str, _: User = Depends(need("researcher"))) -> dict[str, bool]:
        state.eln.link_twin_run(eid, run_id)
        return {"ok": True}

    @app.get("/api/eln/provenance/{node_type}/{node_id}")
    def eln_prov(node_type: str, node_id: str, _: User = Depends(need("viewer"))) -> dict[str, Any]:
        return state.eln.provenance_graph(node_type, node_id)

    # ---------------------------------------------------------------- admin
    @app.post("/api/admin/backup")
    def make_backup(u: User = Depends(need("admin"))) -> dict[str, str]:
        p = backup.create_backup(state.db, settings.data_dir)
        audit(u, "backup", p.name)
        return {"file": p.name}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    _ = io
    return app
