import sys
import os

# ✅ 1. PYTHONPATH 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ 2. .env 로드: 반드시 import 전에! 도커컨테이너 내부 기준
from dotenv import load_dotenv
load_dotenv(dotenv_path="/app/infra/.env")  # 절대 경로로 명시

# 이후 import는 여기에
import pytest
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
async def test_ping():
    transport = ASGITransport(app=app)  # ✅ 여기서 app을 transport로 감쌌고
    async with LifespanManager(app):    # ✅ lifespan 실행
        async with AsyncClient(transport=transport, base_url="http://test") as ac:  # ✅ 핵심 수정
            response = await ac.get("/api/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}