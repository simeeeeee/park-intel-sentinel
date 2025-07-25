import sys
import os
import pytest
from httpx import AsyncClient
from asgi_lifespan import LifespanManager  # ✅ FastAPI의 lifespan 테스트 지원
from httpx import ASGITransport 

# backend 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app  # FastAPI 앱 불러오기


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_ping():
    transport = ASGITransport(app=app)  # ✅ 여기서 app을 transport로 감쌌고
    async with LifespanManager(app):    # ✅ lifespan 실행
        async with AsyncClient(transport=transport, base_url="http://test") as ac:  # ✅ 핵심 수정
            response = await ac.get("/api/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}