from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)
OK = dict(months_in_business=24, annual_revenue=240000, monthly_cash_flow=7000, existing_monthly_debt=800,
          credit_score=700, amount=40000, term_months=24)


def test_home_health():
    assert c.get("/").status_code == 200 and c.get("/health").json()["chunks"] > 0


def test_cited_answer_and_refusal():
    r = c.post("/ask", json={"question": "What credit score do I need?"}).json()
    assert r["grounded"] and r["citations"][0]["heading"] == "Eligibility"
    assert not c.post("/ask", json={"question": "What is the weather in Paris?"}).json()["grounded"]


def test_eligibility_pass_and_fail():
    assert c.post("/api/eligibility", json=OK).json()["verdict"] == "likely_eligible"
    r = c.post("/api/eligibility", json={**OK, "amount": 250000, "credit_score": 580}).json()
    assert r["verdict"] != "likely_eligible" and r["tips"]


def test_vector_map():
    d = c.get("/api/vector", params={"q": "credit score"}).json()
    assert d["hits"] and d["map"]["query"]


def test_insights():
    d = c.get("/api/insights").json()
    assert d["total"] == 60 and len(d["ideas"]) == 3 and d["synthetic"]


def test_offers_process_pipeline_cache():
    r = c.post("/api/eligibility", json=OK).json()
    assert r["offers"][0]["name"] == "Autopay discount" and "0.25" in r["offers"][0]["text"]
    assert len(r["process_steps"]) == 5 and r["payment_after_offers"] < r["monthly_payment"]
    assert len(c.get("/api/pipeline").json()["stages"]) == 7
    q = {"question": "Which documents are required?"}
    c.post("/ask", json=q)
    assert c.post("/ask", json=q).json()["cached"] is True
