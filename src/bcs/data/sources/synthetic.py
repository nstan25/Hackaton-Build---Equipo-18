"""Fuente sintética (stub de T00): sirve las fixtures del repo, sin red ni datos reales."""

from __future__ import annotations

from collections.abc import Iterable

from bcs.data.schema import LabelRow, load_labels
from bcs.paths import fixtures_dir


class SyntheticSource:
    source_id = "synthetic"

    def fetch(self, sample: int | None = None) -> None:
        return None

    def adapt(self) -> Iterable[LabelRow]:
        df = load_labels(fixtures_dir() / "labels_fixture.csv")
        rows = df.to_dict(orient="records")
        return [LabelRow(**{str(k): v for k, v in r.items()}) for r in rows]
