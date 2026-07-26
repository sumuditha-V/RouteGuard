"""API smoke tests. Cover everything except /recommend (which needs GEMINI_API_KEY).

Uses FastAPI's TestClient, so no server needs to be running.
"""

from fastapi.testclient import TestClient

from routeguard.api import app


def test_api_endpoints():
    with TestClient(app) as client:
        # health
        assert client.get("/health").json() == {"status": "ok"}

        # model info
        info = client.get("/model/info").json()
        assert info["version"] == "model_v1"
        assert 0.0 < info["threshold"] < 1.0

        # metrics carry the reliability curve for the dashboard
        m = client.get("/metrics").json()
        assert "reliability_curve" in m
        assert m["brier_calibrated"] < m["brier_raw"]

        # a real sample order -> predict on it
        rec = client.get("/sample", params={"risk": "high"}).json()["record"]
        r = client.post("/predict", json={"record": rec}).json()
        assert 0.0 <= r["probability"] <= 1.0
        assert r["prediction"] in ("late", "on_time")
        assert len(r["shap_drivers"]) > 0
