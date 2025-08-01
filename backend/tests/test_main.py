import sys
import os

# ✅ 1. PYTHONPATH 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ 2. .env 로드
from dotenv import load_dotenv
load_dotenv(dotenv_path="/app/.env")

# ✅ 3. 핵심 모듈 import
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
@patch("app.main.database.connect", new_callable=AsyncMock)
@patch("app.main.database.disconnect", new_callable=AsyncMock)
async def test_ping(mock_disconnect, mock_connect):
    transport = ASGITransport(app=app)
    async with LifespanManager(app):  # ✅ lifespan 실행 → DB 연결은 mock됨
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}
