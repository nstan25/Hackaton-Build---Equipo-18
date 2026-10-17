"""Token compartido (SPEC §2.2): header `X-App-Token` o `?token=`.

No es autenticación de usuarios; es lo mínimo razonable para una demo.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, Query


def require_token(
    x_app_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    expected = os.environ.get("BCS_APP_TOKEN")
    given = x_app_token or token
    # En bytes: compare_digest con str no ASCII levanta TypeError (sería un 500, no un 401).
    if not expected or not given or not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="token inválido o ausente")
