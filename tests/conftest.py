import os
import tempfile
from pathlib import Path


TEST_DB = Path(tempfile.gettempdir()) / "toxic_chat_api_tests.db"
if TEST_DB.exists():
    TEST_DB.unlink()

os.environ.update(
    {
        "APP_ENV": "testing",
        "DATABASE_URL": f"sqlite:///{TEST_DB.as_posix()}",
        "SECRET_KEY": "test-secret-key-with-more-than-32-characters",
        "ALLOW_MOCK_MODEL": "true",
        "REQUIRE_MODEL_MANIFEST": "false",
        "STORE_RAW_MESSAGES": "false",
        "AUTO_BAN_WARNING_COUNT": "0",
    }
)

import pytest
from fastapi.testclient import TestClient

from backend_api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    username = f"user_{os.urandom(5).hex()}"
    response = client.post(
        "/api/auth/register", json={"username": username, "password": "StrongPass123"}
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
