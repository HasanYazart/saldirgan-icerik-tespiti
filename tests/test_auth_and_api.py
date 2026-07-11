from sqlalchemy import select

from backend_api import models
from backend_api.crypto import decrypt_text
from backend_api.database import SessionLocal
from backend_api.security import create_access_token


def test_protected_endpoint_requires_token(client):
    response = client.post("/api/chat/send", json={"text": "merhaba"})
    assert response.status_code == 401


def test_registration_and_login(client):
    username = "login_test_user"
    password = "StrongPass123"
    registered = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert registered.status_code in {201, 409}
    logged_in = client.post(
        "/api/auth/login", data={"username": username, "password": password}
    )
    assert logged_in.status_code == 200
    assert logged_in.json()["token_type"] == "bearer"
    bad_login = client.post(
        "/api/auth/login", data={"username": username, "password": "WrongPassword123"}
    )
    assert bad_login.status_code == 401
    duplicate = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert duplicate.status_code == 409


def test_toxic_message_is_censored_and_encrypted(client, auth_headers):
    response = client.post(
        "/api/chat/send", json={"text": "siktir git"}, headers=auth_headers
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "toxic"
    assert "gizlenmiştir" in payload["displayed_text"]

    with SessionLocal() as db:
        record = db.get(models.MessageLog, payload["message_id"])
        assert record.original_text is None
        assert record.encrypted_text != "siktir git"
        assert decrypt_text(record.encrypted_text) == "siktir git"


def test_client_cannot_choose_another_user(client, auth_headers):
    response = client.post(
        "/api/chat/send",
        json={"text": "temiz bir mesaj", "user_id": 999, "username": "admin"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    with SessionLocal() as db:
        record = db.get(models.MessageLog, response.json()["message_id"])
        token_status = client.get("/api/me/status", headers=auth_headers).json()
        assert record.user_id == token_status["user_id"]


def test_admin_endpoint_rejects_normal_user(client, auth_headers):
    with SessionLocal() as db:
        any_user = db.scalar(select(models.User).order_by(models.User.id.desc()))
    response = client.post(
        f"/api/admin/users/{any_user.id}/reset", headers=auth_headers
    )
    assert response.status_code == 403


def test_blank_message_is_rejected(client, auth_headers):
    response = client.post("/api/chat/send", json={"text": "   "}, headers=auth_headers)
    assert response.status_code == 422


def test_appeal_flow_rejects_duplicates(client, auth_headers):
    message = client.post(
        "/api/chat/send", json={"text": "siktir git"}, headers=auth_headers
    ).json()
    payload = {"message_id": message["message_id"], "reason": "Kararın bağlam nedeniyle yanlış olduğunu düşünüyorum."}
    first = client.post("/api/appeals", json=payload, headers=auth_headers)
    assert first.status_code == 201
    duplicate = client.post("/api/appeals", json=payload, headers=auth_headers)
    assert duplicate.status_code == 409


def test_admin_review_and_reset_flow(client, auth_headers, monkeypatch):
    status = client.get("/api/me/status", headers=auth_headers).json()
    with SessionLocal() as db:
        user = db.get(models.User, status["user_id"])
        assert user is not None
        user.role = "admin"
        db.commit()
        admin_token = create_access_token(user)
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    from backend_api.main import ml_service

    monkeypatch.setattr(
        ml_service,
        "analyze_text",
        lambda text, user_id=0: {
            "is_toxic": False,
            "needs_review": True,
            "toxicity_score": 0.76,
            "detection_method": "bert_calibrated",
            "category": "offensive_language",
            "severity": "medium",
            "model_version": "test-model",
        },
    )
    queued = client.post(
        "/api/chat/send", json={"text": "Bağlama göre belirsiz mesaj"}, headers=admin_headers
    )
    assert queued.json()["status"] == "review"
    records = client.get("/api/admin/reviews", headers=admin_headers)
    assert records.status_code == 200
    assert any(item["id"] == queued.json()["message_id"] for item in records.json())

    decision = client.post(
        f"/api/admin/reviews/{queued.json()['message_id']}",
        json={"decision": "rejected", "admin_note": "Bağlam temiz olarak değerlendirildi."},
        headers=admin_headers,
    )
    assert decision.status_code == 200
    reset = client.post(f"/api/admin/users/{status['user_id']}/reset", headers=admin_headers)
    assert reset.status_code == 200


def test_health_endpoints(client):
    health = client.get("/api/health")
    ready = client.get("/api/ready")
    assert health.status_code == 200
    assert health.json()["status"] in {"online", "degraded"}
    assert ready.status_code == 200
