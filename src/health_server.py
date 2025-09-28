import asyncio
import contextlib
import json
import logging
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger("agent-health")

StatusProvider = Callable[[], Awaitable[Dict[str, object]]]


def _encode_response(status_line: str, body: Dict[str, object]) -> bytes:
    payload = json.dumps(body).encode("utf-8")
    headers = "\r\n".join(
        [
            status_line,
            "Content-Type: application/json",
            f"Content-Length: {len(payload)}",
            "Connection: close",
            "",
            "",
        ]
    ).encode("ascii")
    return headers + payload


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    status_provider: StatusProvider,
) -> None:
    try:
        data = await reader.read(1024)
        if not data:
            return

        request_line = data.split(b"\r\n", 1)[0].decode("utf-8", "ignore")
        parts = request_line.strip().split()
        if len(parts) < 2:
            writer.write(_encode_response("HTTP/1.1 400 Bad Request", {"detail": "Invalid request"}))
            await writer.drain()
            return

        method, path = parts[0], parts[1]
        if method.upper() != "GET":
            writer.write(_encode_response("HTTP/1.1 405 Method Not Allowed", {"detail": "GET only"}))
            await writer.drain()
            return

        status = await status_provider()

        if path == "/health":
            body = status
        elif path == "/version":
            body = {
                "service": status.get("service", "agent"),
                "version": status.get("version", "unknown"),
                "status": status.get("status", "unknown"),
            }
        else:
            writer.write(_encode_response("HTTP/1.1 404 Not Found", {"detail": "Not found"}))
            await writer.drain()
            return

        writer.write(_encode_response("HTTP/1.1 200 OK", body))
        await writer.drain()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Health probe handling failed: %s", exc)
        with contextlib.suppress(Exception):
            writer.write(_encode_response("HTTP/1.1 500 Internal Server Error", {"detail": "probe failure"}))
            await writer.drain()
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def start_health_server(
    *,
    status_provider: StatusProvider,
    host: str,
    port: int,
) -> Optional[asyncio.AbstractServer]:
    try:
        server = await asyncio.start_server(
            lambda r, w: _handle_client(r, w, status_provider),
            host,
            port,
        )
        return server
    except OSError as exc:
        logger.error("Unable to bind health server on %s:%s (%s)", host, port, exc)
        return None


async def stop_health_server(server: asyncio.AbstractServer) -> None:
    server.close()
    await server.wait_closed()
    logger.info("Health endpoint stopped")
