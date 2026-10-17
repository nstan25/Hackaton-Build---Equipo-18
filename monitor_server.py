"""Panel en vivo del entrenamiento (detector YOLO + CNN BCS).

Uso: python monitor_server.py [puerto]  -> http://127.0.0.1:8765
"""
import csv
import json
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent
YOLO_CSV = ROOT / "runs/cow/results.csv"
YOLO_EPOCHS = 20


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def data():
    cnn = {p.stem.removeprefix("train_log_"): read_csv(p) for p in sorted((ROOT / "results").glob("train_log_*.csv"))}
    logs = [YOLO_CSV, *(ROOT / "results").glob("train_log_*.csv")]
    age = {p.stem: time.time() - p.stat().st_mtime for p in logs if p.exists()}
    return {"yolo": read_csv(YOLO_CSV), "yolo_epochs": YOLO_EPOCHS, "cnn": cnn,
            "yolo_age": age.get(YOLO_CSV.stem), "cnn_age": min((v for k, v in age.items() if k != YOLO_CSV.stem), default=None)}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data"):
            body = json.dumps(data()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.path = "/monitor.html"
            super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    HTTPServer(("127.0.0.1", port), lambda *a: Handler(*a, directory=str(ROOT))).serve_forever()
