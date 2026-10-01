"""
Verification script for live running FastAPI server.
Performs end-to-end sample requests against:
- GET /
- POST /predict
- GET /customers
- POST /intervene
- PATCH /intervene/{id}
- GET /customers/{customer_id}
- GET /interventions
"""

import sys
import time
import json
import httpx

BASE_URL = "http://127.0.0.1:8000"


def verify_live_api():
    print("=" * 80)
    print(" LIVE API VERIFICATION: TELCO CHURN PREDICTION & RETENTION BACKEND")
    print("=" * 80)

    # 1. Check Server Liveness (retry up to 5 times)
    client = httpx.Client(base_url=BASE_URL, timeout=10.0)
    connected = False
    for attempt in range(1, 6):
        try:
            res = client.get("/")
            if res.status_code == 200:
                connected = True
                print(f"[OK] Health check passed on attempt {attempt}:")
                print(json.dumps(res.json(), indent=2))
                break
        except Exception:
            time.sleep(1)

    if not connected:
        print("[FAIL] Could not connect to FastAPI server at", BASE_URL)
        sys.exit(1)

    # 1b. Test POST /auth/login
    print("\n" + "-" * 80)
    print(" 0. POST /auth/login (Corporate Email Authentication)")
    print("-" * 80)
    auth_res = client.post("/auth/login", json={"email": "analyst@churniq.ai", "password": "demo"})
    print(f"Status Code: {auth_res.status_code}")
    auth_data = auth_res.json()
    print(f"Auth Success: User={auth_data['user']['name']} ({auth_data['user']['role']}) | Token={auth_data['access_token'][:15]}...")
    assert auth_res.status_code == 200
    assert "access_token" in auth_data

    # 2. Test POST /predict (High-Risk Customer Sample)
    print("\n" + "-" * 80)
    print(" 1. POST /predict (High-Risk Profile)")
    print("-" * 80)
    sample_high_risk = {
        "customer_id": "DEMO-CUST-8810",
        "tenure": 3.0,
        "monthly_charges": 92.40,
        "total_charges": 277.20,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "support_tickets": 4,
        "last_login_days_ago": 32.0,
        "usage_frequency": 11.5
    }
    pred_res = client.post("/predict", json=sample_high_risk)
    print(f"Status Code: {pred_res.status_code}")
    pred_data = pred_res.json()
    print(json.dumps(pred_data, indent=2))
    assert pred_res.status_code == 200
    assert pred_data["predicted_churn"] == 1
    assert len(pred_data["top_risk_factors"]) > 0
    assert len(pred_data["recommended_interventions"]) > 0

    # 3. Test POST /predict (Low-Risk Customer Sample)
    print("\n" + "-" * 80)
    print(" 2. POST /predict (Low-Risk Profile)")
    print("-" * 80)
    sample_low_risk = {
        "customer_id": "DEMO-CUST-8820",
        "tenure": 48.0,
        "monthly_charges": 42.00,
        "total_charges": 2016.00,
        "contract_type": "Two year",
        "payment_method": "Bank transfer (automatic)",
        "internet_service": "DSL",
        "support_tickets": 0,
        "last_login_days_ago": 4.0,
        "usage_frequency": 75.0
    }
    pred_low_res = client.post("/predict", json=sample_low_risk)
    print(f"Status Code: {pred_low_res.status_code}")
    pred_low_data = pred_low_res.json()
    print(json.dumps(pred_low_data, indent=2))
    assert pred_low_res.status_code == 200
    assert pred_low_data["predicted_churn"] == 0

    # 4. Test GET /customers
    print("\n" + "-" * 80)
    print(" 3. GET /customers (List Stored Customers with Scores)")
    print("-" * 80)
    cust_res = client.get("/customers?limit=5")
    print(f"Status Code: {cust_res.status_code}")
    cust_data = cust_res.json()
    print(f"Total Persisted Customers: {cust_data['total_count']}")
    for c in cust_data["customers"][:3]:
        print(f"  - {c['customer_id']:<18} | Churn Prob: {c['churn_probability']:.1%} | Tier: {c['risk_tier']:<18} | Tickets: {c['support_tickets']}")
    assert cust_res.status_code == 200
    assert cust_data["total_count"] >= 2

    # 5. Test POST /intervene
    print("\n" + "-" * 80)
    print(" 4. POST /intervene (Log Intervention for Customer)")
    print("-" * 80)
    intervene_payload = {
        "customer_id": "DEMO-CUST-8810",
        "action_type": "Proactive Support Outreach",
        "status": "Pending",
        "notes": "Assigned to Tier-2 retention specialist due to 4 open tickets."
    }
    int_res = client.post("/intervene", json=intervene_payload)
    print(f"Status Code: {int_res.status_code}")
    int_data = int_res.json()
    print(json.dumps(int_data, indent=2))
    assert int_res.status_code == 201
    assert int_data["status"] == "Pending"
    intervention_id = int_data["id"]

    # 6. Test PATCH /intervene/{id} (Update status to Sent and then Resolved)
    print("\n" + "-" * 80)
    print(f" 5. PATCH /intervene/{intervention_id} (Update Status to 'Sent' and 'Resolved')")
    print("-" * 80)
    patch_res = client.patch(
        f"/intervene/{intervention_id}",
        json={"status": "Sent", "notes": "Outreach phone call scheduled and retention email sent."}
    )
    print(f"Status Code: {patch_res.status_code}")
    print(json.dumps(patch_res.json(), indent=2))
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "Sent"

    patch_res2 = client.patch(
        f"/intervene/{intervention_id}",
        json={"status": "Resolved", "notes": "Customer accepted 15% annual plan renewal."}
    )
    assert patch_res2.status_code == 200
    assert patch_res2.json()["status"] == "Resolved"

    # 7. Test GET /customers/{customer_id}
    print("\n" + "-" * 80)
    print(" 6. GET /customers/DEMO-CUST-8810 (Fetch Customer + Intervention History)")
    print("-" * 80)
    detail_res = client.get("/customers/DEMO-CUST-8810")
    print(f"Status Code: {detail_res.status_code}")
    detail_data = detail_res.json()
    print(f"Customer: {detail_data['customer_id']}")
    print(f"Churn Prob: {detail_data['churn_probability']:.1%}")
    print(f"Interventions History Count: {len(detail_data['interventions_history'])}")
    for h in detail_data["interventions_history"]:
        print(f"  - [{h['status']}] {h['action_type']} (Updated: {h['updated_at']})")
    assert detail_res.status_code == 200
    assert len(detail_data["interventions_history"]) >= 1

    # 8. Test GET /interventions
    print("\n" + "-" * 80)
    print(" 7. GET /interventions (All Logged Interventions)")
    print("-" * 80)
    all_int_res = client.get("/interventions")
    print(f"Status Code: {all_int_res.status_code}")
    all_int_data = all_int_res.json()
    print(f"Total Interventions Logged: {len(all_int_data)}")
    for record in all_int_data[:3]:
        print(f"  #{record['id']} | Customer: {record['customer_id']} | Action: {record['action_type']} | Status: {record['status']}")
    assert all_int_res.status_code == 200

    print("\n" + "=" * 80)
    print(" ALL LIVE ENDPOINT CHECKS VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    verify_live_api()
