"""G3: un BCS en el extremo de la grilla del modelo es `low_confidence` (SPEC §1.2)."""

from __future__ import annotations

import pytest

from bcs.contract import QualityReport, decide_status, is_edge

GOOD = QualityReport(is_cow_rear=True, score=0.9, issues=[])
STEP = 0.25
EDGES = (3.25, 4.25)  # grilla dairy_sdb


@pytest.mark.parametrize(
    "bcs, expected",
    [(3.25, "low_confidence"), (3.5, "ok"), (3.75, "ok"), (4.0, "ok"), (4.25, "low_confidence")],
)
def test_edge_class_is_low_confidence(bcs: float, expected: str) -> None:
    status = decide_status(GOOD, 0.95, (3.25, 3.75), STEP, bcs=bcs, edges=EDGES)
    assert status == expected


def test_edges_optional_keeps_previous_rules() -> None:
    assert decide_status(GOOD, 0.95, (3.0, 3.5), STEP, bcs=3.25) == "ok"
    assert decide_status(GOOD, 0.95, (3.0, 3.5), STEP) == "ok"


def test_rejected_wins_over_edge() -> None:
    bad = QualityReport(is_cow_rear=False, score=0.1, issues=["no parece una vaca de atrás"])
    assert decide_status(bad, 0.95, (3.25, 3.5), STEP, bcs=3.25, edges=EDGES) == "rejected"


def test_is_edge_tolerance() -> None:
    assert is_edge(3.25, EDGES, STEP) and is_edge(4.25, EDGES, STEP)
    assert not is_edge(3.5, EDGES, STEP) and not is_edge(4.0, EDGES, STEP)
