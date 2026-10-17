"""`make smoke` (SPEC §3.2): levanta la API con una base temporal y la prueba de punta a punta."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from bcs.contract import validate_prediction_dict
from bcs.scale import load_scale

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
TOKEN = "smoke-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(c: httpx.Client, name: str, cow_id: str | None, **kw) -> httpx.Response:
    data = {"cow_id": cow_id} if cow_id is not None else {}
    files = {"image": (f"{name}.jpg", (FIXTURES / f"{name}.jpg").read_bytes(), "image/jpeg")}
    return c.post("/predict", data=data, files=files, **kw)


def run(base: str) -> list[str]:
    scale = load_scale()
    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    with httpx.Client(base_url=base, headers={"X-App-Token": TOKEN}, timeout=30) as c:
        check(c.get("/health", headers={"X-App-Token": ""}).status_code == 200, "/health sin token")
        sent: dict[str, list[str]] = {}
        for name, cow in [("rear_cow", "uy-001"), ("side_cow", "UY-001"), ("dog", " uy-002 ")]:
            r = _post(c, name, cow)
            ok = r.status_code == 200
            errors = validate_prediction_dict(r.json(), scale.keys, scale.step) if ok else []
            check(ok and not errors, f"POST /predict {name} → contrato {errors or ''}")
            if ok:
                body = r.json()
                check(body["cow_id"] == cow.strip().upper(), f"{name}: cow_id normalizado")
                sent.setdefault(body["cow_id"], []).append(body["prediction_id"])
        check(_post(c, "rear_cow", None).status_code == 422, "sin cow_id → 422")
        check(_post(c, "rear_cow", "uy 001").status_code == 422, "cow_id inválido → 422")
        check(
            _post(c, "rear_cow", "UY-001", headers={"X-App-Token": ""}).status_code == 401,
            "sin token → 401",
        )
        check(
            c.get("/cows", headers={"X-App-Token": ""}).status_code == 401, "/cows sin token → 401"
        )

        cows = {row["cow_id"]: row for row in c.get("/cows").json()}
        check(set(cows) == set(sent), f"/cows refleja {sorted(sent)}")
        for cow_id, ids in sent.items():
            row = cows.get(cow_id, {})
            check(row.get("n_evaluations") == len(ids), f"/cows {cow_id}: {len(ids)} evaluaciones")
            check(row.get("last_prediction_id") == ids[-1], f"/cows {cow_id}: última foto")
            hist = c.get(f"/cows/{cow_id}").json()
            check(
                sorted(p["prediction_id"] for p in hist) == sorted(ids), f"/cows/{cow_id} historial"
            )
            r = c.get(f"/predictions/{ids[-1]}")
            check(
                r.status_code == 200 and r.json()["cow_id"] == cow_id,
                f"/predictions/{{id}} {cow_id}",
            )
            r = c.get(f"/media/{ids[-1]}.jpg")
            check(
                r.status_code == 200 and r.content[:2] == b"\xff\xd8", f"/media/{{id}}.jpg {cow_id}"
            )
        check(c.get("/", params={"token": TOKEN}).status_code == 200, "web/ servido")
    return fails


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bcs-smoke-") as tmp:
        port = _free_port()
        env = {
            **os.environ,
            "BCS_APP_TOKEN": TOKEN,
            "BCS_DB_PATH": f"{tmp}/bcs.sqlite",
            "BCS_MEDIA_DIR": f"{tmp}/media",
        }
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "bcs.api:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]
        proc = subprocess.Popen(cmd, env=env)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                try:
                    if httpx.get(f"{base}/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if proc.poll() is not None:
                    print("FAIL la API no arrancó", file=sys.stderr)
                    return 1
                time.sleep(0.2)
            fails = run(base)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    print(f"\nsmoke: {'OK' if not fails else f'{len(fails)} fallas'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
