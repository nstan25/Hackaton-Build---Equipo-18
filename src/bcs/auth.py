"""Token compartido (SPEC §2.2): header `X-App-Token`, `?token=` o la cookie `bcs_token`.

No es autenticación de usuarios; es lo mínimo razonable para una demo. La web lee el `?token=` del
link compartido, lo guarda (localStorage + cookie `SameSite=Strict`) y lo saca de la barra de
direcciones; las llamadas a la API van con el header y las páginas con la cookie (R5).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, MutableMapping
from http.cookies import CookieError, SimpleCookie
from typing import Any
from urllib.parse import parse_qs, unquote

from fastapi import Cookie, Header, HTTPException, Query

COOKIE = "bcs_token"
# Lo que acepta `POST /predict` antes de leerlo: la foto (`api_predict.MAX_IMAGE_BYTES`) + el resto
# del form y el multipart. Más que esto se corta sin guardar nada (R6).
MAX_PREDICT_BODY = 21 * 1024 * 1024

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def token_ok(given: str | None) -> bool:
    expected = os.environ.get("BCS_APP_TOKEN")
    # En bytes: compare_digest con str no ASCII levanta TypeError (sería un 500, no un 401).
    return bool(expected and given and hmac.compare_digest(given.encode(), expected.encode()))


def require_token(
    x_app_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
    bcs_token: str | None = Cookie(default=None),
) -> None:
    # la web guarda la cookie con encodeURIComponent
    if not token_ok(x_app_token or token or (bcs_token and unquote(bcs_token))):
        raise HTTPException(status_code=401, detail="token inválido o ausente")


def _scope_token(scope: Scope) -> str | None:
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
    if headers.get("x-app-token"):
        return headers["x-app-token"]
    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    if query.get("token") and query["token"][0]:
        return query["token"][0]
    try:
        morsel = SimpleCookie(headers.get("cookie", "")).get(COOKIE)
    except CookieError:
        return None
    return unquote(morsel.value) if morsel is not None else None


class _BodyTooLarge(Exception):
    pass


class PredictGuard:
    """Middleware ASGI de `POST /predict`: token y tamaño **antes** de leer el cuerpo (R6).

    FastAPI parsea el multipart (y Starlette lo vuelca a un temporal en disco) antes de resolver
    `Depends(require_token)`; acá se corta antes. Sin `Content-Length` (chunked), se cuentan los
    bytes al recibirlos y se corta al pasar `MAX_PREDICT_BODY`.
    """

    def __init__(self, app: ASGIApp, max_body: int | None = None) -> None:
        self.app = app
        self._max_body = max_body

    @property
    def max_body(self) -> int:
        return self._max_body if self._max_body is not None else MAX_PREDICT_BODY

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/predict":
            await self.app(scope, receive, send)
            return
        if not token_ok(_scope_token(scope)):
            await _reply(send, 401, "token inválido o ausente")
            return
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_body):
            await _reply(send, 413, f"el cuerpo supera {self.max_body // 2**20} MB")
            return

        seen = 0
        too_large = replied = False

        async def counted_receive() -> Message:
            nonlocal seen, too_large
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > self.max_body:
                too_large = True
                raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # FastAPI convierte el error al parsear el form en un 400: se responde 413 en su lugar.
            nonlocal replied
            if not too_large:
                await send(message)
            elif not replied:
                replied = True
                await _reply(send, 413, f"el cuerpo supera {self.max_body // 2**20} MB")

        try:
            await self.app(scope, counted_receive, guarded_send)
        except _BodyTooLarge:
            if not replied:
                await _reply(send, 413, f"el cuerpo supera {self.max_body // 2**20} MB")


async def _reply(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


_TOKEN_IN_URL = re.compile(r"([?&]token=)[^&\s\"]*")


def redact(text: str) -> str:
    return _TOKEN_IN_URL.sub(r"\1***", text)


class RedactTokenFilter(logging.Filter):
    """Reemplaza `token=...` por `token=***` en los logs (el access log de uvicorn tiene la URL)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


def install_log_redaction(logger_name: str = "uvicorn.access") -> None:
    logger = logging.getLogger(logger_name)
    if not any(isinstance(f, RedactTokenFilter) for f in logger.filters):
        logger.addFilter(RedactTokenFilter())
