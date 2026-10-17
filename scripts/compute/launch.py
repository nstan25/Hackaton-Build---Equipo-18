"""`make train MODEL=<m> BACKEND=<local|modal|kaggle|tenki> [FOLDS=all] [SMOKE=1]` (SPEC §2.4).

Cada backend solo empaqueta, lanza `python -m bcs.train`, trae `report.json` a
`runs/<task>/<ts>/` y los pesos a `$BCS_ARTIFACTS_DIR/<task>/`, y anota la corrida en
`runs/compute_ledger.csv`.

Datos: si hay `labels.csv` en `$BCS_DATA_DIR`, los remotos usan el pack de `make pack-data`
(sin holdout). Si no hay y es `--smoke`, corren sobre las fixtures sintéticas del repo.

Credenciales: ningún código las lee; solo se mira si existen. Si falta la de un backend, sale
con código 3 y lo dice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
import pack_data  # noqa: E402

from bcs.paths import REPO_ROOT, artifacts_dir, config_dir, data_dir, runs_dir  # noqa: E402
from bcs.train import FOLD_ALIASES, SINGLE_FOLD, resolve_config_path, run  # noqa: E402

HERE = Path(__file__).resolve().parent
EXIT_UNAVAILABLE = 3
KAGGLE_DATASET = "bcs-pack"
TENKI_WORKFLOW = "cpu-job.yml"


class Unavailable(RuntimeError):
    """El backend no está configurado (sin CLI o sin credencial)."""


class KernelTimeout(RuntimeError):
    """Se venció la espera de un job remoto que puede seguir corriendo (y gastando)."""


# ---------------------------------------------------------------- comunes


def train_args(model: str, task: str, fold: str, smoke: bool) -> list[str]:
    """Argumentos de `bcs.train` comunes a todos los backends (config relativo al repo)."""
    cfg = resolve_config_path(model)
    args = ["--config", str(cfg.relative_to(REPO_ROOT)), "--fold", fold, "--task", task]
    if cfg == config_dir() / "train.yaml":
        args += ["--model", model]
    if smoke:
        args.append("--smoke")
    return args


def parse_folds(folds: str | None) -> list[str]:
    """G0: un único split train/val. `all`/`val`/`0` (o nada) → ["val"]."""
    if folds in (None, "") or folds in FOLD_ALIASES:
        return [SINGLE_FOLD]
    raise SystemExit(f"FOLDS={folds!r}: hay un único split train/val (G0); usá all o nada")


def maybe_pack(smoke: bool) -> Path | None:
    """Pack para los remotos; None si no hay labels.csv y es smoke (fixtures sintéticas)."""
    if not (data_dir() / "labels.csv").exists():
        if smoke:
            print("train: sin labels.csv → smoke con fixtures sintéticas", file=sys.stderr)
            return None
        raise SystemExit(f"train: no existe {data_dir() / 'labels.csv'} (make labels)")
    return pack_data.build_pack()


def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, text=True, **kw)


def _need_cli(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise Unavailable(f"no está la CLI `{name}` en el PATH")
    return path


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    return out.stdout.strip() or "unknown"


def collect(src: Path, task: str, ts: str, fold: str) -> Path:
    """Trae la salida remota de `bcs.train` (report.json, config.json, PNG, predicciones, pesos):
    todo a runs/<task>/<ts>_<fold>/ y pesos + predicciones también a artifacts/<task>/."""
    report_src = next(src.rglob("report.json"), None)
    if report_src is None:
        raise RuntimeError(f"la corrida remota no dejó report.json (mirá {src})")
    out = runs_dir() / task / f"{ts}_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    art = artifacts_dir() / task
    art.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_src.read_text())
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        shutil.copy2(f, out / f.name)
        if f.suffix == ".weights" or f.name.startswith("predictions_"):
            shutil.copy2(f, art / f.name)
            key = "weights" if f.suffix == ".weights" else "predictions"
            report["train"][key] = str(art / f.name)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------- local


def backend_local(model: str, task: str, folds: list[str], smoke: bool) -> list[Path]:
    paths = []
    for fold in folds:
        cfg = resolve_config_path(model)
        report = run(
            cfg,
            fold=fold,
            smoke=smoke,
            model=model if cfg == config_dir() / "train.yaml" else None,
            task=task,
            backend="local",
        )
        paths.append(Path(report["report_path"]))
    return paths


# ---------------------------------------------------------------- modal


def _modal_ready() -> str:
    cli = _need_cli("modal")
    has_env = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    if not has_env and not (Path.home() / ".modal.toml").exists():
        raise Unavailable("sin credencial de Modal (`modal token new` lo hace el humano)")
    return cli


def backend_modal(model: str, task: str, folds: list[str], smoke: bool) -> list[Path]:
    cli = _modal_ready()
    pack = maybe_pack(smoke)
    vol = os.environ.get("BCS_MODAL_VOLUME", "bcs-data")
    if pack is not None:
        _run([cli, "volume", "create", vol], capture_output=True)
        ls = _run([cli, "volume", "ls", vol, "/packs"], capture_output=True)
        if pack.name not in ls.stdout:
            _run([cli, "volume", "put", vol, str(pack), f"/packs/{pack.name}"], check=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory(prefix="bcs-modal-") as tmp:
        spec = {
            "folds": folds,
            "train_args": {f: train_args(model, task, f, smoke) for f in folds},
            "pack": pack.name if pack else "",
            "task": task,
            "git_sha": _git_sha(),
            "out": tmp,
        }
        _run(
            [cli, "run", f"{HERE / 'modal_app.py'}::main", "--spec", json.dumps(spec)],
            check=True,
            cwd=REPO_ROOT,
        )
        return [collect(Path(tmp) / fold, task, ts, fold) for fold in folds]


# ---------------------------------------------------------------- kaggle


def _kaggle_user() -> str:
    """Usuario de Kaggle: `--kaggle-user` o `KAGGLE_USERNAME`. Nunca se lee la credencial; solo
    se mira que exista (env `KAGGLE_KEY` o el archivo que puso el humano)."""
    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        raise Unavailable("falta el usuario de Kaggle: --kaggle-user o KAGGLE_USERNAME")
    has_cred = bool(os.environ.get("KAGGLE_KEY")) or any(
        p.exists()
        for p in (Path.home() / ".kaggle/kaggle.json", Path.home() / ".config/kaggle/kaggle.json")
    )
    if not has_cred:
        raise Unavailable("sin credencial de Kaggle (~/.kaggle/kaggle.json lo pone el humano)")
    return user


def kaggle_push_pack(cli: str, user: str, pack: Path) -> str:
    """Sube el pack como dataset privado `<user>/bcs-pack` (create o version). Devuelve el ref."""
    ref = f"{user}/{KAGGLE_DATASET}"
    with tempfile.TemporaryDirectory(prefix="bcs-kds-") as tmp:
        d = Path(tmp)
        try:
            os.link(pack, d / pack.name)
        except OSError:
            shutil.copy2(pack, d / pack.name)
        meta = {"title": KAGGLE_DATASET, "id": ref, "licenses": [{"name": "other"}]}
        (d / "dataset-metadata.json").write_text(json.dumps(meta))
        files = _run([cli, "datasets", "files", ref], capture_output=True)
        if files.returncode != 0:
            _run([cli, "datasets", "create", "-p", tmp], check=True)  # privado por defecto
        elif pack.name not in files.stdout:
            _run([cli, "datasets", "version", "-p", tmp, "-m", pack.stem], check=True)
    for _ in range(120):
        st = _run([cli, "datasets", "status", ref], capture_output=True)
        if "ready" in st.stdout.lower():
            break
        time.sleep(10)
    return ref


def _payload_b64() -> str:
    import base64
    import io

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in ("src/bcs", "config", "tests/fixtures"):
            tar.add(
                REPO_ROOT / rel,
                arcname=rel,
                filter=lambda ti: None if "__pycache__" in ti.name else ti,
            )
    return base64.b64encode(buf.getvalue()).decode()


def kaggle_stage_kernel(
    d: Path, user: str, slug: str, dataset: str | None, args: dict[str, Any]
) -> None:
    meta = json.loads((HERE / "kaggle" / "kernel-metadata.json").read_text())
    meta["id"] = f"{user}/{slug}"
    meta["title"] = slug
    meta["dataset_sources"] = [dataset] if dataset else []
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    body = (HERE / "kaggle" / "kernel_main.py").read_text()
    escaped = json.dumps(args).replace("\\", "\\\\").replace('"', '\\"')
    body = body.replace('PAYLOAD = "__BCS_PAYLOAD__"', f'PAYLOAD = "{_payload_b64()}"')
    body = body.replace('ARGS = json.loads("__BCS_ARGS__")', f'ARGS = json.loads("{escaped}")')
    (d / meta["code_file"]).write_text(body)


def backend_kaggle(model: str, task: str, folds: list[str], smoke: bool) -> list[Path]:
    cli = _need_cli("kaggle")
    user = _kaggle_user()
    pack = maybe_pack(smoke)
    dataset = kaggle_push_pack(cli, user, pack) if pack else None
    slug = re.sub(r"[^a-z0-9-]+", "-", f"bcs-train-{task}".lower())
    kernel = f"{user}/{slug}"
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timeout_s = int(os.environ.get("BCS_KAGGLE_TIMEOUT_S", 12 * 3600))
    paths = []
    for fold in folds:
        args = {
            "train_args": train_args(model, task, fold, smoke),
            "pack": pack.name if pack else "",
            "git_sha": _git_sha(),
        }
        with tempfile.TemporaryDirectory(prefix="bcs-kk-") as tmp:
            kaggle_stage_kernel(Path(tmp), user, slug, dataset, args)
            _run([cli, "kernels", "push", "-p", tmp], check=True)
        t0 = time.monotonic()
        status = ""
        while time.monotonic() - t0 < timeout_s:
            time.sleep(30)
            status = _run([cli, "kernels", "status", kernel], capture_output=True).stdout.lower()
            if any(s in status for s in ("complete", "error", "cancel")):
                break
        if not any(s in status for s in ("complete", "error", "cancel")):
            raise KernelTimeout(
                f"kernel {kernel}: sin terminar a los {timeout_s} s; PUEDE SEGUIR CORRIENDO en "
                f"Kaggle (mirá `kaggle kernels status {kernel}`)"
            )
        if "complete" not in status:
            raise RuntimeError(
                f"kernel {kernel}: {status.strip()} (log: `kaggle kernels output {kernel}`)"
            )
        with tempfile.TemporaryDirectory(prefix="bcs-ko-") as out:
            _run([cli, "kernels", "output", kernel, "-p", out], check=True)
            paths.append(collect(Path(out), task, ts, fold))
    return paths


# ---------------------------------------------------------------- tenki


def _tenki_ready(cli: str) -> None:
    st = _run([cli, "auth", "status"], capture_output=True)
    if st.returncode != 0:
        raise Unavailable("gh no está autenticado")
    wf = _run(
        [cli, "api", f"repos/{{owner}}/{{repo}}/actions/workflows/{TENKI_WORKFLOW}"],
        capture_output=True,
    )
    if wf.returncode != 0:
        raise Unavailable(
            f".github/workflows/{TENKI_WORKFLOW} no está en el remoto (falta mergear y pushear)"
        )


def backend_tenki(model: str, task: str, folds: list[str], smoke: bool) -> list[Path]:
    cli = _need_cli("gh")
    _tenki_ready(cli)
    pack_source = "none"
    if (data_dir() / "labels.csv").exists():
        # El runner baja el pack del dataset privado de Kaggle (secrets KAGGLE_*).
        kcli = _need_cli("kaggle")
        kaggle_push_pack(kcli, _kaggle_user(), pack_data.build_pack())
        pack_source = "kaggle"
    elif not smoke:
        raise SystemExit(f"train: no existe {data_dir() / 'labels.csv'} (make labels)")
    ref = os.environ.get("BCS_TENKI_REF", "main")
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    paths = []
    for fold in folds:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        _run(
            [cli, "workflow", "run", TENKI_WORKFLOW, "--ref", ref,
             "-f", f"train_args={' '.join(train_args(model, task, fold, smoke))}",
             "-f", f"task={task}", "-f", f"pack_source={pack_source}"],
            check=True,
        )  # fmt: skip
        run_id = ""
        for _ in range(30):
            time.sleep(5)
            out = _run(
                [cli, "run", "list", "--workflow", TENKI_WORKFLOW, "--event", "workflow_dispatch",
                 "--limit", "5", "--json", "databaseId,createdAt"],
                capture_output=True, check=True,
            )  # fmt: skip
            runs = [r for r in json.loads(out.stdout) if r["createdAt"] >= started[:19]]
            if runs:
                run_id = str(runs[0]["databaseId"])
                break
        if not run_id:
            raise RuntimeError("no encontré la corrida de cpu-job.yml recién lanzada")
        _run([cli, "run", "watch", run_id, "--exit-status"], check=True)
        with tempfile.TemporaryDirectory(prefix="bcs-gh-") as out:
            _run([cli, "run", "download", run_id, "-D", out], check=True)
            paths.append(collect(Path(out), task, ts, fold))
    return paths


# ---------------------------------------------------------------- main

BACKENDS = {
    "local": backend_local,
    "modal": backend_modal,
    "kaggle": backend_kaggle,
    "tenki": backend_tenki,
}
MACHINE = {
    "local": "local",
    "modal": f"modal:{os.environ.get('BCS_MODAL_GPU', 'L4')}",
    "kaggle": "kaggle",
    "tenki": "tenki",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    p.add_argument("--folds", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--task", default=None, help="por defecto $BCS_TASK o el modelo")
    p.add_argument("--kaggle-user", default=None, help="por defecto $KAGGLE_USERNAME")
    a = p.parse_args(argv)
    task = a.task or os.environ.get("BCS_TASK") or a.model
    folds = parse_folds(a.folds)
    if a.kaggle_user:
        os.environ["KAGGLE_USERNAME"] = a.kaggle_user
    t0 = time.monotonic()

    def fail(status: str) -> None:
        ledger.append(
            backend=a.backend, task=task, model=a.model, fold=",".join(folds), smoke=a.smoke,
            machine=MACHINE[a.backend], seconds=time.monotonic() - t0, status=status,
        )  # fmt: skip

    try:
        paths = BACKENDS[a.backend](a.model, task, folds, a.smoke)
        wall = time.monotonic() - t0
        for path in paths:
            report = json.loads(path.read_text())["train"]
            # local: tiempo de pared; remotos: lo que corrió el contenedor (lo que se cobra).
            seconds = (
                wall / len(paths) if a.backend == "local" else float(report.get("seconds", wall))
            )
            row = ledger.append(
                backend=a.backend, task=task, model=a.model, fold=report["fold"], smoke=a.smoke,
                machine=f"{MACHINE[a.backend]}:{report.get('device', '?')}",
                seconds=seconds, status="ok", report=str(path),
            )  # fmt: skip
            print(
                f"train[{a.backend}]: {path} · {row['seconds']} s · ~${row['est_cost_usd']}",
                file=sys.stderr,
            )
            print(path)
    except Unavailable as e:  # no arrancó: no va a la bitácora
        print(f"train[{a.backend}]: no disponible: {e}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except KernelTimeout as e:
        fail("failed:timeout_may_still_run")
        print(f"train[{a.backend}]: falló: {e}", file=sys.stderr)
        return 1
    except BaseException as e:  # toda corrida que arrancó deja su línea, también Ctrl-C
        fail("failed")
        print(f"train[{a.backend}]: falló: {type(e).__name__}: {e}", file=sys.stderr)
        if isinstance(e, Exception):
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
