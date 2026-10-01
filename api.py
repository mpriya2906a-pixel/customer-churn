"""
FastAPI Backend Application for Customer Churn Prediction & Retention Management.
Exposes endpoints:
- POST /predict: Computes churn probability & SHAP risk factors, recommends interventions, and persists customer score.
- GET /customers: Returns stored customer profiles and churn scores.
- POST /intervene: Logs recommended or actioned interventions (Pending/Sent/Responded/Resolved).
- GET /customers/{customer_id}: Detailed customer profile and intervention timeline.
- GET /interventions: Complete audit log of all interventions.
- PATCH /intervene/{intervention_id}: Updates an intervention's lifecycle status.
"""

import os
import json
import uuid
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.database import (
    init_db,
    save_or_update_customer,
    get_customer,
    get_customers,
    count_customers,
    log_intervention,
    get_customer_interventions,
    get_all_interventions,
    update_intervention_status,
    get_dashboard_stats,
    DEFAULT_DB_PATH
)
from src.recommender import InterventionRecommender
from src.schemas import (
    CustomerInput,
    PredictionResponse,
    InterventionCreate,
    InterventionResponse,
    InterventionStatusUpdate,
    CustomerListResponse,
    LoginRequest,
    LoginResponse,
    UserProfile
)
from src.explainability import ChurnExplainer, load_explainer



# Lifespan context manager for loading artifacts on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    model_path = os.path.join(models_dir, "best_model.joblib")
    explainer_path = os.path.join(models_dir, "shap_explainer.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    # Initialize DB
    init_db()

    # Verify and load ML artifacts
    if not (os.path.exists(prep_path) and os.path.exists(model_path) and os.path.exists(explainer_path)):
        raise RuntimeError(
            f"Required ML artifacts not found in {models_dir}. "
            "Please run 'python main.py' to generate and serialize models and explainers."
        )

    print("[API Lifespan] Loading preprocessor, best model, and SHAP explainer...")
    app.state.preprocessor = joblib.load(prep_path)
    app.state.model = joblib.load(model_path)
    app.state.explainer = load_explainer(explainer_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    app.state.metadata = metadata

    print(f"[API Lifespan] Successfully loaded model: {metadata.get('best_model_name', 'Production Model')}")
    yield


app = FastAPI(
    title="Customer Churn & Behaviour Prediction API",
    description=(
        "Production REST API for real-time customer churn scoring, SHAP explainability, "
        "rule-based intervention recommendations, and SQLite-backed retention logging."
    ),
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health & Info"])
async def root():
    """
    API health status and model metadata endpoint.
    """
    return {
        "service": "Customer Churn Prediction & Retention API",
        "status": "online",
        "version": "2.0.0",
        "active_model": getattr(app.state, "metadata", {}).get("best_model_name", "Logistic Regression"),
        "endpoints": {
            "predict": "POST /predict",
            "customers": "GET /customers",
            "intervene": "POST /intervene",
            "documentation": "/docs"
        }
    }


@app.get("/stats", tags=["Dashboard"])
def get_stats():
    """
    Returns summary statistics for the dashboard.
    """
    try:
        return get_dashboard_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Preconfigured authorized corporate demo users
DEFAULT_USERS = {
    "admin@churniq.ai": {
        "name": "Sarah Connor",
        "role": "Chief Retention Officer",
        "avatar_initials": "SC"
    },
    "analyst@churniq.ai": {
        "name": "Alex Morgan",
        "role": "Senior Churn Analyst",
        "avatar_initials": "AM"
    },
    "csm@churniq.ai": {
        "name": "David Chen",
        "role": "Customer Success Lead",
        "avatar_initials": "DC"
    }
}


@app.post(
    "/auth/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    tags=["Authentication"],
    summary="User email authentication for dashboard access"
)
async def login_user(payload: LoginRequest):
    """
    Authenticates a user via their corporate email address.
    Supports predefined demo profiles (admin@churniq.ai, analyst@churniq.ai, csm@churniq.ai)
    as well as any valid corporate email address for frictionless sign-in.
    """
    email = payload.email.strip().lower()

    # Basic email validation
    if "@" not in email or "." not in email.split("@")[-1] or len(email) < 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Please provide a valid corporate email address."
        )

    # Check against known users or generate profile dynamically
    if email in DEFAULT_USERS:
        profile_data = DEFAULT_USERS[email]
    else:
        local_part = email.split("@")[0]
        # Generate friendly display name from email (e.g. john.doe -> John Doe)
        name_parts = [part.capitalize() for part in local_part.replace(".", " ").replace("_", " ").split()]
        display_name = " ".join(name_parts) if name_parts else "Team Member"
        initials = "".join([p[0].upper() for p in name_parts[:2]]) if name_parts else "TM"
        profile_data = {
            "name": display_name,
            "role": "Retention Specialist",
            "avatar_initials": initials
        }

    user_profile = UserProfile(
        email=email,
        name=profile_data["name"],
        role=profile_data["role"],
        avatar_initials=profile_data["avatar_initials"]
    )

    # Generate synthetic session token
    token = f"churniq_tok_{uuid.uuid4().hex[:16]}"

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user=user_profile,
        message=f"Welcome back, {user_profile.name}!"
    )



@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions & Explainability"],
    summary="Predict churn probability and extract SHAP risk factors"
)
async def predict_churn(customer: CustomerInput):
    """
    Accepts customer profile attributes, generates churn probability using the best production model,
    calculates top local SHAP risk factors and protective factors, maps them to automated
    intervention recommendations, and persists the customer & score to SQLite.
    """
    try:
        raw_dict = customer.model_dump()
        customer_df = pd.DataFrame([raw_dict])

        # Preprocess features
        preprocessor = app.state.preprocessor
        model = app.state.model
        explainer: ChurnExplainer = app.state.explainer

        X_processed = preprocessor.transform(customer_df)

        # Compute SHAP explanation and probabilities
        explanation = explainer.explain_instance(
            customer_processed_row=X_processed,
            customer_raw_data=raw_dict,
            top_k=4
        )

        # Rule-based intervention recommender
        recommendations = InterventionRecommender.recommend(
            customer_data=raw_dict,
            top_risk_factors=explanation["top_risk_factors"],
            churn_probability=explanation["churn_probability"]
        )

        # Persist customer record and churn score to SQLite
        save_or_update_customer(
            customer_raw=raw_dict,
            prediction_result=explanation,
            recommended_interventions=recommendations
        )

        return PredictionResponse(
            customer_id=explanation["customer_id"],
            churn_probability=explanation["churn_probability"],
            predicted_churn=explanation["predicted_churn"],
            risk_tier=explanation["risk_tier"],
            top_risk_factors=explanation["top_risk_factors"],
            top_protective_factors=explanation["top_protective_factors"],
            recommended_interventions=recommendations,
            persisted=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get(
    "/customers",
    response_model=CustomerListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Customers"],
    summary="List stored customers and their churn scores"
)
async def list_customers(
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    risk_tier: Optional[str] = Query(default=None, description="Filter by risk tier (e.g. 'CRITICAL RISK')"),
    min_churn_probability: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Filter customers above minimum churn probability threshold"
    )
):
    """
    Returns stored customer records, churn probabilities, risk tiers, and latest intervention status.
    """
    customers = get_customers(
        limit=limit,
        offset=offset,
        risk_tier=risk_tier,
        min_churn_probability=min_churn_probability
    )
    total_count = count_customers(
        risk_tier=risk_tier,
        min_churn_probability=min_churn_probability
    )

    return CustomerListResponse(
        total_count=total_count,
        limit=limit,
        offset=offset,
        customers=customers
    )


@app.get(
    "/customers/{customer_id}",
    status_code=status.HTTP_200_OK,
    tags=["Customers"],
    summary="Retrieve customer profile, churn score, and intervention history"
)
async def get_single_customer(customer_id: str):
    """
    Retrieves full customer profile, predicted churn metrics, and complete timeline of interventions.
    """
    customer = get_customer(customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with id '{customer_id}' not found in database."
        )

    interventions = get_customer_interventions(customer_id)
    customer["interventions_history"] = interventions
    return customer


@app.post(
    "/intervene",
    response_model=InterventionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Interventions"],
    summary="Log a recommended or actioned intervention for a customer"
)
async def create_intervention(intervention: InterventionCreate):
    """
    Logs an intervention (Pending, Sent, Responded, Resolved) for a customer in the database.
    """
    try:
        logged_record = log_intervention(
            customer_id=intervention.customer_id,
            action_type=intervention.action_type,
            status=intervention.status,
            notes=intervention.notes
        )
        return InterventionResponse(**logged_record)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")


@app.get(
    "/interventions",
    response_model=List[InterventionResponse],
    status_code=status.HTTP_200_OK,
    tags=["Interventions"],
    summary="List all logged interventions"
)
async def list_interventions(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status")
):
    """
    Returns audit log of all logged interventions across the customer base.
    """
    records = get_all_interventions(limit=limit, offset=offset, status=status_filter)
    return [InterventionResponse(**r) for r in records]


@app.patch(
    "/intervene/{intervention_id}",
    response_model=InterventionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Interventions"],
    summary="Update the status or notes of an existing intervention"
)
async def patch_intervention(intervention_id: int, payload: InterventionStatusUpdate):
    """
    Updates an intervention's lifecycle status (e.g. from Pending to Sent, or Sent to Resolved).
    """
    updated = update_intervention_status(
        intervention_id=intervention_id,
        status=payload.status,
        notes=payload.notes
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Intervention with id {intervention_id} not found."
        )
    return InterventionResponse(**updated)

# Mount the static dashboard last to avoid intercepting API routes
dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")
