import asyncio
import importlib
import sys
from pathlib import Path
from typing import AsyncIterator, Tuple

import httpx
from httpx import ASGITransport
import jwt
import pytest
from fakeredis.aioredis import FakeRedis

# Ensure MLMSOLUTIONINFRA is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
async def app_with_fake_redis(monkeypatch) -> AsyncIterator[Tuple[object, FakeRedis]]:
    env = {
        "LIVEKIT_WS_URL": "wss://example.livekit.dev",
        "LIVEKIT_API_KEY": "lkc_test",
        "LIVEKIT_API_SECRET": "lks_test_secret",
        "DEFAULT_ALLOWED_EMBED_ORIGINS": "https://example.com,https://fallback.io",
        "WIDGET_PUBLIC_BASE": "https://voice.test",
        "TOKEN_TTL_SECONDS": "180",
        "RATE_LIMIT_WINDOW_SECONDS": "60",
        "RATE_LIMIT_MAX_REQUESTS": "250",
        "REDIS_URL": "redis://redis:6379/5",
        "ADMIN_TOKEN": "test-admin-token",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    for module_name in list(sys.modules):
        if module_name.startswith("MLMSOLUTIONINFRA.voice_widget"):
            sys.modules.pop(module_name)

    module = importlib.import_module("MLMSOLUTIONINFRA.voice_widget.app_embed")
    fake = FakeRedis(decode_responses=True)

    async def _get_fake():
        return fake

    module.app.dependency_overrides[module._get_redis] = _get_fake
    module.app.state.redis = fake

    try:
        yield module, fake
    finally:
        module.app.dependency_overrides.clear()
        await fake.flushall()
        module.app.state.redis = None


async def seed_tenant(fake: FakeRedis, tenant_id: str, **fields) -> None:
    payload = {
        "allowed_origins": fields.get("allowed_origins", "https://example.com"),
        "default_agent": fields.get("default_agent", "voice-alpha"),
        "status": fields.get("status", "active"),
    }
    await fake.hset(f"tenant:{tenant_id}", mapping=payload)


@pytest.mark.asyncio
async def test_token_issued_for_allowed_origin(app_with_fake_redis) -> None:
    module, fake = app_with_fake_redis
    await seed_tenant(fake, "tenantA")

    transport = ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get(
            "/api/livekit/token",
            params={"tenant": "tenantA", "room": "mlm-tenantA", "identity": "web-1"},
            headers={"X-Embed-Origin": "https://example.com"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ws_url"] == module.settings.livekit_ws_url

    decoded = jwt.decode(
        data["token"],
        module.settings.livekit_api_secret,
        algorithms=["HS256"],
        issuer=module.settings.livekit_api_key,
        options={"require": ["iss", "sub", "exp", "nbf"]},
    )
    assert decoded["sub"] == "web-1"
    assert decoded["video"]["room"]["name"] == "mlm-tenantA"
    assert decoded["exp"] - decoded["nbf"] <= module.settings.token_ttl_seconds


@pytest.mark.asyncio
async def test_token_rejected_for_disallowed_origin(app_with_fake_redis) -> None:
    module, fake = app_with_fake_redis
    await seed_tenant(fake, "tenantB", allowed_origins="https://partner.io")

    transport = ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get(
            "/api/livekit/token",
            params={"tenant": "tenantB", "room": "mlm-tenantB", "identity": "web-2"},
            headers={"X-Embed-Origin": "https://example.com"},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_csp_header_reflects_allowed_origins(app_with_fake_redis) -> None:
    module, fake = app_with_fake_redis
    await seed_tenant(fake, "tenantC", allowed_origins="https://embed.me,https://partner.io")

    transport = ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/widget", params={"tenant": "tenantC", "agent": "alpha"})
    assert resp.status_code == 200
    csp = resp.headers.get("content-security-policy")
    assert csp is not None
    assert "frame-ancestors 'self' https://embed.me https://partner.io" in csp


@pytest.mark.asyncio
async def test_rate_limit_enforced(app_with_fake_redis) -> None:
    module, fake = app_with_fake_redis
    await seed_tenant(fake, "tenantRate", allowed_origins="https://example.com")

    module.settings.rate_limit_max_requests = 3
    module.settings.rate_limit_window_seconds = 60

    transport = ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        headers = {"X-Embed-Origin": "https://example.com"}
        params = {"tenant": "tenantRate", "room": "room", "identity": "user-1"}
        for _ in range(3):
            ok_resp = await client.get("/api/livekit/token", params=params, headers=headers)
            assert ok_resp.status_code == 200
        blocked = await client.get("/api/livekit/token", params=params, headers=headers)
    assert blocked.status_code == 429


@pytest.mark.asyncio
async def test_snippet_contains_iframe(app_with_fake_redis) -> None:
    module, fake = app_with_fake_redis
    await seed_tenant(fake, "tenantSnippet", allowed_origins="https://example.com", default_agent="beta")

    transport = ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get(
            "/api/embed/snippet", params={"tenant": "tenantSnippet"}, headers={"X-Admin-Token": "test-admin-token"}
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert "<iframe" in payload["snippet"]
    assert "allow=\"microphone; autoplay; clipboard-write\"" in payload["snippet"]
    assert "width:360px" in payload["snippet"]


@pytest.mark.asyncio
async def test_parallel_token_requests_succeed(app_with_fake_redis) -> None:
    module, fake = app_with_fake_redis
    await seed_tenant(fake, "tenantLoad", allowed_origins="https://example.com")

    transport = ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        headers = {"X-Embed-Origin": "https://example.com"}

        async def hit(idx: int) -> int:
            params = {"tenant": "tenantLoad", "room": "room", "identity": f"user-{idx}"}
            resp = await client.get("/api/livekit/token", params=params, headers=headers)
            return resp.status_code

        statuses = await asyncio.gather(*[hit(i) for i in range(100)])

    assert all(code == 200 for code in statuses)
