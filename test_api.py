"""
Unit and Integration Tests for FastAPI Churn Backend API.
Tests:
- /predict endpoint with high and low risk profiles
- /customers endpoint with filtering and pagination
- /intervene endpoint lifecycle (Pending -> Sent -> Resolved)
- Rule-based intervention recommender logic
- Input validation and error handling
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.recommender import InterventionRecommender
from src.database import init_db, get_db_connection


@pytest.fixture(scope="session")
def client():
    """
    TestClient fixture that triggers app lifespan (loading models and initializing DB).
    """
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """
    Verify GET / returns API metadata and online status.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "predict" in data["endpoints"]
    assert "active_model" in data


def test_predict_high_risk_customer(client):
    """
    Verify /predict accurately identifies high risk customer,
    generates SHAP factors, recommendations, and persists record.
    """
    payload = {
        "customer_id": "TEST-CUST-HIGH-01",
        "tenure": 2.0,
        "monthly_charges": 98.50,
        "total_charges": 197.00,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "support_tickets": 5,
        "last_login_days_ago": 35.0,
        "usage_frequency": 8.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "TEST-CUST-HIGH-01"
    assert data["churn_probability"] >= 0.70
    assert data["predicted_churn"] == 1
    assert data["risk_tier"] == "CRITICAL RISK"
    assert len(data["top_risk_factors"]) > 0
    assert "feature" in data["top_risk_factors"][0]
    assert "shap_value" in data["top_risk_factors"][0]
    assert data["top_risk_factors"][0]["shap_value"] > 0

    # Verify recommendations
    actions = [r["action_type"] for r in data["recommended_interventions"]]
    assert "Proactive Support Outreach" in actions
    assert "Contract Upgrade Incentive" in actions


def test_predict_low_risk_customer(client):
    """
    Verify /predict accurately scores low risk customer.
    """
    payload = {
        "customer_id": "TEST-CUST-LOW-02",
        "tenure": 60.0,
        "monthly_charges": 35.00,
        "total_charges": 2100.00,
        "contract_type": "Two year",
        "payment_method": "Credit card (automatic)",
        "internet_service": "DSL",
        "support_tickets": 0,
        "last_login_days_ago": 2.0,
        "usage_frequency": 65.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "TEST-CUST-LOW-02"
    assert data["churn_probability"] < 0.35
    assert data["predicted_churn"] == 0
    assert data["risk_tier"] == "LOW RISK (LOYAL)"
    assert len(data["top_protective_factors"]) > 0


def test_predict_validation_error(client):
    """
    Verify validation fails for invalid contract type and negative values.
    """
    invalid_payload = {
        "customer_id": "TEST-INVALID",
        "tenure": -5.0,  # Negative tenure
        "monthly_charges": 50.0,
        "contract_type": "Three year",  # Invalid contract type
        "payment_method": "Electronic check",
        "internet_service": "DSL",
        "support_tickets": 0,
        "last_login_days_ago": 5,
        "usage_frequency": 10
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_customers_list_and_filter(client):
    """
    Verify /customers returns persisted customers and filters by risk_tier.
    """
    response = client.get("/customers")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] >= 2
    customer_ids = [c["customer_id"] for c in data["customers"]]
    assert "TEST-CUST-HIGH-01" in customer_ids

    # Test filtering by risk tier
    filter_resp = client.get("/customers?risk_tier=CRITICAL%20RISK")
    assert filter_resp.status_code == 200
    filtered_data = filter_resp.json()
    for cust in filtered_data["customers"]:
        assert cust["risk_tier"] == "CRITICAL RISK"


def test_single_customer_detail(client):
    """
    Verify /customers/{customer_id} returns full record and 404 for unknown customer.
    """
    response = client.get("/customers/TEST-CUST-HIGH-01")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST-CUST-HIGH-01"
    assert "top_risk_factors" in data
    assert "interventions_history" in data

    # 404 test
    not_found = client.get("/customers/NON-EXISTENT-ID")
    assert not_found.status_code == 404


def test_intervene_full_lifecycle(client):
    """
    Verify /intervene logs interventions, checks valid statuses, and supports status updates.
    """
    # 1. Log intervention with status='Pending'
    inv_payload = {
        "customer_id": "TEST-CUST-HIGH-01",
        "action_type": "Proactive Support Outreach",
        "status": "Pending",
        "notes": "Initiated ticket review for high-churn customer."
    }
    create_resp = client.post("/intervene", json=inv_payload)
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["customer_id"] == "TEST-CUST-HIGH-01"
    assert created["status"] == "Pending"
    inv_id = created["id"]

    # 2. Update status to 'Sent'
    patch_resp = client.patch(f"/intervene/{inv_id}", json={"status": "Sent", "notes": "Email dispatched."})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "Sent"

    # 3. Update status to 'Resolved'
    patch_resp2 = client.patch(f"/intervene/{inv_id}", json={"status": "Resolved", "notes": "Customer retained."})
    assert patch_resp2.status_code == 200
    assert patch_resp2.json()["status"] == "Resolved"

    # 4. Check interventions audit list
    audit_resp = client.get("/interventions")
    assert audit_resp.status_code == 200
    all_invs = audit_resp.json()
    assert any(i["id"] == inv_id for i in all_invs)

    # 5. Invalid status test
    invalid_inv = {
        "customer_id": "TEST-CUST-HIGH-01",
        "action_type": "Discount",
        "status": "NonExistentStatus"
    }
    bad_resp = client.post("/intervene", json=invalid_inv)
    assert bad_resp.status_code == 422


def test_recommender_rules():
    """
    Unit test for InterventionRecommender logic across varying attributes.
    """
    # Test high tickets rule
    rec1 = InterventionRecommender.recommend(
        customer_data={"support_tickets": 4, "contract_type": "Two year"},
        top_risk_factors=[],
        churn_probability=0.7
    )
    action_types1 = [r["action_type"] for r in rec1]
    assert "Proactive Support Outreach" in action_types1

    # Test month-to-month rule
    rec2 = InterventionRecommender.recommend(
        customer_data={"support_tickets": 0, "contract_type": "Month-to-month"},
        top_risk_factors=[{"feature": "contract_type_Two year", "shap_value": 0.5}],
        churn_probability=0.6
    )
    action_types2 = [r["action_type"] for r in rec2]
    assert "Contract Upgrade Incentive" in action_types2

    # Test inactivity rule
    rec3 = InterventionRecommender.recommend(
        customer_data={"last_login_days_ago": 30.0, "usage_frequency": 5.0},
        top_risk_factors=[],
        churn_probability=0.55
    )
    action_types3 = [r["action_type"] for r in rec3]
    assert "Re-engagement & Adoption Campaign" in action_types3


def test_auth_login_predefined_user(client):
    """
    Verify /auth/login returns valid session token and profile for demo user.
    """
    payload = {
        "email": "analyst@churniq.ai",
        "password": "demo-password-123",
        "remember_me": True
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "analyst@churniq.ai"
    assert data["user"]["name"] == "Alex Morgan"
    assert data["user"]["role"] == "Senior Churn Analyst"
    assert data["user"]["avatar_initials"] == "AM"


def test_auth_login_dynamic_corporate_user(client):
    """
    Verify /auth/login handles generic corporate email sign-ins dynamically.
    """
    payload = {
        "email": "jordan.belfort@enterprise.com",
        "password": "secure-password",
        "remember_me": False
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "jordan.belfort@enterprise.com"
    assert data["user"]["name"] == "Jordan Belfort"
    assert data["user"]["avatar_initials"] == "JB"


def test_auth_login_invalid_email(client):
    """
    Verify /auth/login rejects invalid email formats.
    """
    bad_payload = {
        "email": "not-an-email",
        "password": "123"
    }
    response = client.post("/auth/login", json=bad_payload)
    assert response.status_code == 422

