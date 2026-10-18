"""`make browser` (SPEC §3.2): la UI a 390 px contra la API dummy, en Firefox headless.

Maneja Firefox por WebDriver (geckodriver) con `httpx`, sin dependencias nuevas. Firefox no achica
la ventana a menos de 500 px, así que la app corre dentro de un iframe de 390 × 844 (`/__harness`):
los media queries y el layout ven 390 px. Las capturas quedan en `runs/T07/`.

Recorrido: dos vacas con IDs distintos (una foto ok, una low_confidence, el perro rechazado y otra
ok), el rodeo con las dos y su miniatura, filtros y búsqueda, y el historial de cada una con solo
sus fotos.

`--model` (T12): la misma recorrida contra el modelo de producción (`BCS_PREDICTOR=model`). Las
fotos salen en runtime de val de `$BCS_DATA_DIR` (nunca test, nunca se commitean), con semilla fija:
una de clase 3.75 que da `ok` y una de clase 3.25 que da `low_confidence` en el extremo (G3). Lo que
muestra la UI se compara con la `Prediction` que guardó la API. Capturas en `runs/T12/browser/`.
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
TOKEN = "browser-token"
ELEMENT = "element-6066-11e4-a52e-4f735466cecf"
WIDTH, HEIGHT = 390, 844
SEED = 42
PICK_TRIES = 25


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, proc: subprocess.Popen[bytes], what: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=2)
            return
        except httpx.TransportError:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"{what} terminó antes de arrancar")
        time.sleep(0.2)
    raise RuntimeError(f"{what} no respondió en {timeout:.0f} s")


class WebDriverError(RuntimeError):
    pass


class Browser:
    """Cliente W3C WebDriver mínimo."""

    def __init__(self, base: str) -> None:
        self.http = httpx.Client(base_url=base, timeout=60)
        caps = {"moz:firefoxOptions": {"args": ["-headless", "--width=600", "--height=1000"]}}
        self.sid = self._call("POST", "/session", {"capabilities": {"alwaysMatch": caps}})[
            "sessionId"
        ]

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        r = self.http.request(method, path, json=body)
        value = r.json().get("value")
        if r.status_code >= 400:
            raise WebDriverError(f"{path}: {value.get('error')}: {value.get('message')}")
        return value

    def cmd(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._call(method, f"/session/{self.sid}{path}", body)

    def close(self) -> None:
        try:
            self.cmd("DELETE", "")
        finally:
            self.http.close()

    def go(self, url: str) -> None:
        self.cmd("POST", "/url", {"url": url})

    def js(self, script: str, *args: Any) -> Any:
        return self.cmd("POST", "/execute/sync", {"script": script, "args": list(args)})

    def find(self, css: str) -> dict[str, str]:
        return {
            ELEMENT: self.cmd("POST", "/element", {"using": "css selector", "value": css})[ELEMENT]
        }

    def click(self, css: str) -> None:
        # Si la página re-renderiza entre el find y el click, el elemento queda viejo: se reintenta.
        for attempt in range(3):
            try:
                self.cmd("POST", f"/element/{self.find(css)[ELEMENT]}/click", {})
                return
            except WebDriverError as e:
                if "stale element" not in str(e) or attempt == 2:
                    raise
                time.sleep(0.2)

    def type(self, css: str, text: str) -> None:
        el = self.find(css)[ELEMENT]
        self.cmd("POST", f"/element/{el}/clear", {})
        self.cmd("POST", f"/element/{el}/value", {"text": text})

    def frame(self, el: dict[str, str] | None) -> None:
        self.cmd("POST", "/frame", {"id": el})

    def wait(self, script: str, what: str, timeout: float = 15) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            value = self.js(script)
            if value:
                return value
            if time.monotonic() > deadline:
                raise AssertionError(f"timeout esperando: {what}")
            time.sleep(0.1)


DUMMY_PHOTOS = {
    "central": FIXTURES / "rear_cow.jpg",
    "extreme": FIXTURES / "side_cow.jpg",
    "dog": FIXTURES / "dog.jpg",
}


def pick_val_photos() -> dict[str, Path]:
    """Dos fotos de val (`splits.csv`) para `--model`, con el pipeline de producción en este
    proceso: la primera de clase 3.75 que da `ok` y la primera de clase 3.25 que da
    `low_confidence` con BCS 3.25, en un orden barajado con semilla fija."""
    import pandas as pd

    from bcs import inference
    from bcs.paths import data_dir

    d = data_dir()
    labels = pd.read_csv(d / "labels.csv", usecols=["image_id", "path", "bcs"])
    splits = pd.read_csv(d / "splits.csv")
    val = labels.merge(splits.loc[splits["split"] == "val"], on="image_id")
    pipe = inference.model_pipeline()

    def first(cls: float, want: Any) -> Path:
        rows = val.loc[np.isclose(val["bcs"], cls)].sample(frac=1.0, random_state=SEED)
        for rel in rows["path"].head(PICK_TRIES):
            path = d / rel
            if want(inference.run_pipeline(pipe, path.read_bytes(), "PICK")):
                return path
        raise RuntimeError(f"ninguna foto de val de clase {cls} cumple en {PICK_TRIES} intentos")

    return {
        "central": first(3.75, lambda p: p.status == "ok" and p.bcs > 3.25 + 1e-9),
        "extreme": first(3.25, lambda p: p.status == "low_confidence" and p.bcs == 3.25),
        "dog": FIXTURES / "dog.jpg",
    }


class Run:
    def __init__(
        self, browser: Browser, base: str, out: Path, photos: dict[str, Path], model: bool = False
    ) -> None:
        self.b = browser
        self.base = base
        self.out = out
        self.photos = photos
        self.model = model
        self.fails: list[str] = []
        self.shots: list[str] = []

    def check(self, ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            self.fails.append(what)

    def open(self, page: str) -> None:
        """Abre `page` en el iframe de 390 px y entra a su contexto."""
        self.b.frame(None)
        self.b.js(
            "document.getElementById('app').src = arguments[0];",
            f"{self.base}/{page}?token={TOKEN}",
        )
        self.b.frame(self.b.find("#app"))
        self.b.wait(
            f"return location.pathname.endsWith('/{page}') && document.readyState === 'complete'",
            f"carga de {page}",
        )

    def shot(self, name: str) -> None:
        """Captura la página entera a 390 px de ancho (estira el iframe al alto del contenido)."""
        height = self.b.js("scrollTo(0, 0); return document.documentElement.scrollHeight")
        self.b.frame(None)
        frame = self.b.find("#app")
        self.b.js(f"arguments[0].height = {max(height, HEIGHT)}", frame)
        png = self.b.cmd("GET", f"/element/{frame[ELEMENT]}/screenshot")
        path = self.out / f"{name}.png"
        path.write_bytes(base64.b64decode(png))
        self.shots.append(str(path.relative_to(ROOT)))
        self.b.js(f"arguments[0].height = {HEIGHT}", frame)
        self.b.frame(frame)

    def text(self, css: str) -> str:
        return self.b.js("return document.querySelector(arguments[0]).textContent.trim()", css)

    def visible(self, css: str) -> bool:
        return self.b.js(
            "const e = document.querySelector(arguments[0]);"
            "return !!e && e.getClientRects().length > 0"
            " && getComputedStyle(e).visibility !== 'hidden'",
            css,
        )

    def images_loaded(self, css: str) -> list[bool]:
        self.b.wait(
            f"return [...document.querySelectorAll({css!r})].every(i => i.complete)",
            f"carga de {css}",
        )
        return self.b.js(
            "return [...document.querySelectorAll(arguments[0])].map(i => i.naturalWidth > 0)", css
        )

    def no_hscroll(self, where: str) -> None:
        w = self.b.js("return [document.documentElement.scrollWidth, innerWidth]")
        self.check(w[0] <= w[1], f"{where}: sin scroll horizontal a {w[1]} px ({w[0]})")

    # --- evaluar -----------------------------------------------------------------------------

    def api(self, path: str) -> dict[str, Any]:
        return httpx.get(f"{self.base}{path}", headers={"X-App-Token": TOKEN}, timeout=30).json()

    def shows_prediction(self, pid: str, what: str) -> dict[str, Any]:
        """`--model`: BCS, confianza (texto y barra) e intervalo en pantalla = la `Prediction`."""
        p = self.api(f"/predictions/{pid}")
        conf = int(np.floor(p["confidence"] * 100 + 0.5))  # Math.round de JS
        self.check(self.text("#res-bcs") == f"{p['bcs']:.2f}", f"{what}: BCS {p['bcs']:.2f}")
        self.check(self.text("#res-conf") == f"{conf}%", f"{what}: confianza {conf}%")
        lo, hi = p["interval"]
        self.check(
            self.text("#res-interval") == f"{lo:.2f} – {hi:.2f}", f"{what}: intervalo {lo}–{hi}"
        )
        width = self.b.js(
            "return parseFloat(document.getElementById('res-conf-bar').firstElementChild"
            ".style.width)"
        )
        self.check(abs(width - p["confidence"] * 100) < 0.1, f"{what}: barra de confianza")
        self.check(
            self.b.js("return document.querySelectorAll('#res-dist rect').length")
            >= len(p["distribution"]),
            f"{what}: distribución dibujada",
        )
        return p

    def evaluate(self, cow_typed: str, photo: str, shot: str | None = None) -> dict[str, str]:
        fixture = self.photos[photo].name
        self.b.type("#cow-id", cow_typed)
        self.b.cmd(
            "POST",
            f"/element/{self.b.find('#photo-camera')[ELEMENT]}/value",
            {"text": str(self.photos[photo])},
        )
        self.b.wait("return !document.getElementById('confirm').hidden", "confirmación")
        cow = cow_typed.strip().upper()
        self.check(self.text("#confirm-cow") == f"Vaca: {cow}", f"confirma 'Vaca: {cow}'")
        if shot:
            self.shot(shot + "-confirmar")
        self.b.click("#send")
        self.b.wait(
            "const r = document.getElementById('result');"
            "return !r.hidden && r.dataset.predictionId",
            f"resultado de {fixture}",
            timeout=120 if self.model else 30,  # el modelo puede estar terminando de cargar
        )
        res = self.b.js(
            "const r = document.getElementById('result');"
            "return {id: r.dataset.predictionId, status: r.dataset.status}"
        )
        self.check(self.text("#res-cow") == f"Vaca: {cow}", f"{fixture}: resultado con el ID {cow}")
        self.check(all(self.images_loaded("#res-thumb")), f"{fixture}: miniatura del resultado")
        if shot:
            self.shot(shot + "-resultado")
        return res

    def next_cow(self) -> None:
        self.b.click("#next")
        self.check(
            self.b.js("return document.getElementById('cow-id').value") == ""
            and not self.visible("#result"),
            "'Siguiente vaca' limpia el ID y el resultado",
        )

    def flow(self) -> None:
        self.open("index.html")
        self.check(self.b.js("return innerWidth") == WIDTH, f"viewport de {WIDTH} px")
        self.check(self.visible("#cow-id"), "campo ID de la vaca visible arriba")
        attrs = self.b.js(
            "const i = document.getElementById('photo-camera');"
            "return [i.getAttribute('accept'), i.getAttribute('capture')]"
        )
        self.check(attrs == ["image/*", "environment"], "botón de cámara con capture=environment")
        self.check(
            "no reemplaza la evaluación de un técnico" in self.text("footer"), "aviso de apoyo"
        )
        self.check("3.25–4.25" in self.text("footer"), "rango validado 3.25–4.25")
        self.check(
            "Sin evaluación en fotos de celular" in self.text("footer"),
            "aviso: sin fotos de celular",
        )
        self.check(
            TOKEN not in httpx.get(f"{self.base}/index.html?token={TOKEN}").text
            and TOKEN not in httpx.get(f"{self.base}/rodeo.html?token={TOKEN}").text,
            "el token no está escrito en el HTML",
        )
        self.b.js("document.getElementById('protocol').open = true")
        self.no_hscroll("index")
        self.shot("01-inicio")

        # Sin ID no se puede evaluar.
        self.b.type("#cow-id", " ")
        self.b.cmd(
            "POST",
            f"/element/{self.b.find('#photo-camera')[ELEMENT]}/value",
            {"text": str(FIXTURES / "rear_cow.jpg")},
        )
        self.b.wait("return !document.getElementById('confirm').hidden", "confirmación sin ID")
        self.check(
            self.visible("#id-error")
            and self.b.js("return document.getElementById('send').disabled"),
            "sin ID: error y botón Evaluar deshabilitado",
        )
        self.b.click("#change")

        ids: dict[str, list[str]] = {"UY-101": [], "UY-202": []}
        central = self.photos["central"].name
        r = self.evaluate(" uy-101 ", "central", shot="02")
        ids["UY-101"].append(r["id"])
        self.check(r["status"] == "ok", f"{central} → ok")
        if self.model:
            self.central_bcs = self.shows_prediction(r["id"], central)["bcs"]
        else:
            self.central_bcs = 3.5
            self.check(self.text("#res-bcs") == "3.50", "BCS grande 3.50")
            self.check(self.text("#res-conf") == "92%", "confianza 92%")
            self.check(self.text("#res-interval") == "3.25 – 3.75", "intervalo 3.25 – 3.75")
            self.check(
                self.b.js("return document.querySelectorAll('#res-dist rect').length") >= 17,
                "distribución dibujada",
            )
            self.check(
                self.b.js(
                    "return parseFloat(document.getElementById('res-conf-bar').firstElementChild"
                    ".style.width)"
                )
                == 92,
                "barra de confianza al 92%",
            )
        self.no_hscroll("resultado")
        self.next_cow()

        extreme = self.photos["extreme"].name
        r = self.evaluate("UY-202", "extreme")
        ids["UY-202"].append(r["id"])
        self.check(r["status"] == "low_confidence", f"{extreme} → low_confidence")
        if self.model:
            self.shows_prediction(r["id"], extreme)
        self.check(self.visible("#res-low"), "aviso de confianza baja visible")
        self.check(
            "posible condición baja" in self.text("#res-low"), "low_confidence en 3.25 → verificar"
        )
        self.shot("03-confianza-baja")
        self.next_cow()

        r = self.evaluate("UY-202", "dog")
        ids["UY-202"].append(r["id"])
        self.check(r["status"] == "rejected", "fixture del perro → rejected")
        self.check(
            self.visible("#res-rejected") and not self.visible("#res-score"),
            "rechazo visible, sin BCS",
        )
        self.check(
            "no parece una vaca de atrás" in self.text("#res-issues"), "motivo del rechazo visible"
        )
        self.no_hscroll("rechazo")
        self.shot("04-rechazo")
        self.next_cow()

        r = self.evaluate("uy-101", "central")
        ids["UY-101"].append(r["id"])

        self.open("index.html")
        self.check(
            self.b.js("return document.getElementById('cow-id').value") == "UY-101",
            "recuerda el último ID",
        )
        self.b.wait(
            "return document.querySelectorAll('#cow-list option').length === 2",
            "autocompletado desde /cows",
        )
        self.check(
            sorted(
                self.b.js(
                    "return [...document.querySelectorAll('#cow-list option')].map(o => o.value)"
                )
            )
            == ["UY-101", "UY-202"],
            "autocompletado con las dos vacas",
        )
        self.rodeo(ids)

    # --- rodeo -------------------------------------------------------------------------------

    def cards(self) -> list[str]:
        return self.b.js(
            "return [...document.querySelectorAll('#cows .cow')].map(a => a.dataset.cowId)"
        )

    def toggle(self, css: str) -> None:
        self.b.click(css)

    def rodeo(self, ids: dict[str, list[str]]) -> None:
        self.b.click("#to-rodeo")
        self.b.wait(
            "return location.pathname.endsWith('/rodeo.html')"
            " && document.querySelectorAll('#cows .cow').length",
            "rodeo con vacas",
        )
        self.check(
            self.b.js("return location.search") == f"?token={TOKEN}", "link al rodeo con token"
        )
        self.check(
            self.cards() == ["UY-202", "UY-101"], f"dos vacas, BCS ascendente: {self.cards()}"
        )
        thumbs = self.images_loaded("#cows .cow img")
        self.check(len(thumbs) == 2 and all(thumbs), "las dos vacas con miniatura")
        self.no_hscroll("rodeo")
        self.shot("05-rodeo")

        self.b.type("#q", "101")
        self.check(self.cards() == ["UY-101"], "búsqueda por ID")
        self.b.type("#q", "")
        self.toggle("#f-rej")
        self.check(self.cards() == ["UY-202"], "filtro 'última foto rechazada'")
        self.shot("06-rodeo-filtro-rechazada")
        self.toggle("#f-rej")
        self.toggle("#f-low")
        self.check(self.cards() == ["UY-202"], "filtro 'baja confianza'")
        self.toggle("#f-low")
        self.toggle("#f-bcs")
        self.check(self.cards() == ["UY-202"], "filtro BCS ≤ 3.25")
        self.b.type("#f-bcs-max", f"{self.central_bcs:g}")
        self.check(self.cards() == ["UY-202", "UY-101"], f"filtro BCS ≤ {self.central_bcs:g}")
        self.toggle("#f-bcs")

        for cow, n_valid, shot in (
            ("UY-101", 2, "07-historial-uy101"),
            ("UY-202", 1, "08-historial-uy202"),
        ):
            self.b.click(f'#cows .cow[data-cow-id="{cow}"]')
            self.b.wait(
                f"return document.querySelectorAll('#hist .hist-item').length === {len(ids[cow])}",
                f"historial de {cow}",
            )
            shown = self.b.js(
                "return [...document.querySelectorAll('#hist .hist-item')]"
                ".map(li => li.dataset.predictionId)"
            )
            self.check(sorted(shown) == sorted(ids[cow]), f"historial de {cow}: solo sus fotos")
            self.check(self.text("#hist-cow") == f"Vaca {cow}", f"historial titulado {cow}")
            imgs = self.images_loaded("#hist .hist-item img")
            self.check(all(imgs), f"historial de {cow}: fotos cargadas")
            self.check(
                self.b.js("return document.querySelectorAll('#chart circle').length") == n_valid,
                f"gráfico SVG de {cow} con {n_valid} puntos válidos",
            )
            self.no_hscroll(f"historial {cow}")
            self.shot(shot)
            self.b.click("#back")
            self.b.wait(
                "return !document.getElementById('list-view').hidden"
                " && document.querySelectorAll('#cows .cow').length === 2",
                "vuelta al rodeo",
            )

        statuses = self.b.js(
            "return [...document.querySelectorAll('#cows .chip')].map(c => c.textContent)"
        )
        self.check(
            statuses == ["Rechazada", "OK"], f"estado de la última foto en cada tarjeta: {statuses}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="make browser: la UI a 390 px contra la API dummy")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--model", action="store_true", help="contra el modelo de producción, con fotos de val"
    )
    args = parser.parse_args()
    args.out = args.out or ROOT / "runs" / ("T12/browser" if args.model else "T07")
    gecko = shutil.which("geckodriver")
    if gecko is None:
        print("FAIL falta geckodriver (y Firefox) en el PATH", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    model_env = {"BCS_PREDICTOR": "model", "BCS_SCALE": "dairy_sdb"} if args.model else {}
    if args.model:
        os.environ.update(model_env)
        photos = pick_val_photos()
        print("fotos de val:", *(str(p) for p in photos.values()), sep="\n  ")
    else:
        photos = DUMMY_PHOTOS

    with tempfile.TemporaryDirectory(prefix="bcs-browser-") as tmp:
        api_port, wd_port = _free_port(), _free_port()
        env = {
            **os.environ,
            "BCS_APP_TOKEN": TOKEN,
            "BCS_DB_PATH": f"{tmp}/bcs.sqlite",
            "BCS_MEDIA_DIR": f"{tmp}/media",
            **model_env,
        }
        server = Path(__file__).with_name("dummy_server.py")
        api = subprocess.Popen(
            [sys.executable, str(server), str(api_port), *(["--model"] if args.model else [])],
            env=env,
        )
        wd = subprocess.Popen(
            [gecko, "--port", str(wd_port), "--log", "error"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs = [api, wd]
        base = f"http://127.0.0.1:{api_port}"
        browser: Browser | None = None
        run: Run | None = None
        try:
            _wait_http(f"{base}/health", api, "la API dummy")
            _wait_http(f"http://127.0.0.1:{wd_port}/status", wd, "geckodriver")
            browser = Browser(f"http://127.0.0.1:{wd_port}")
            browser.go(f"{base}/__harness")
            run = Run(browser, base, args.out, photos, model=args.model)
            try:
                run.flow()
            except (AssertionError, WebDriverError) as e:
                run.check(False, f"recorrido cortado: {e}")
        finally:
            if browser is not None:
                browser.close()
            for p in procs:
                p.terminate()
                p.wait(timeout=10)

    if run is None:
        return 1
    print("\ncapturas:", *run.shots, sep="\n  ")
    print(f"\nbrowser: {'OK' if not run.fails else f'{len(run.fails)} fallas'}")
    return 1 if run.fails else 0


if __name__ == "__main__":
    sys.exit(main())
