from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_assess_404_patient():
    r = client.post("/assess", json={"patient_id": "DOES-NOT-EXIST"})
    assert r.status_code == 404


def test_chat_404_patient():
    r = client.post("/chat", json={"patient_id": "DOES-NOT-EXIST", "message": "hello"})
    assert r.status_code == 404


def test_history_clear_unknown_patient():
    # depending on your implementation this might be 200 or 404, so make it flexible
    r = client.delete("/history/DOES-NOT-EXIST")
    assert r.status_code in (200, 404)
