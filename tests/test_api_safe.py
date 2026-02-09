from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200


def test_assess_patient_not_found():
    r = client.post("/assess", json={"patient_id": "NOPE"})
    assert r.status_code == 404


def test_chat_patient_not_found():
    r = client.post("/chat", json={"patient_id": "NOPE", "message": "hello"})
    assert r.status_code == 404
