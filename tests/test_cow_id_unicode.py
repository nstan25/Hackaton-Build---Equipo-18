"""R13 (eval/review.md): un cow_id con letras no ASCII no se convierte en otro ID."""

from __future__ import annotations

import pytest

from bcs.contract import InvalidCowId, normalize_cow_id


@pytest.mark.parametrize("raw", ["ß1", "vaca-ñ", "ﬁ12", "İ7", "Ａ12"])
def test_non_ascii_cow_id_is_rejected(raw: str) -> None:
    with pytest.raises(InvalidCowId):
        normalize_cow_id(raw)


def test_ascii_still_normalized() -> None:
    assert normalize_cow_id("  uy-0042 ") == "UY-0042"
