"""
Pydantic Schemas for API Requests, Responses, and Validation.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator
import uuid


class CustomerInput(BaseModel):
    """
    Input schema representing raw customer profile data for churn prediction.
    """
    customer_id: Optional[str] = Field(
        default=None,
        description="Unique customer identifier. Auto-generated if not provided.",
        examples=["CUST-77401"]
    )
    tenure: float = Field(
        ...,
        ge=0,
        description="Number of months customer has stayed with company.",
        examples=[4.0]
    )
    monthly_charges: float = Field(
        ...,
        ge=0,
        description="Monthly bill amount in USD.",
        examples=[89.50]
    )
    total_charges: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total charges to date in USD. If omitted, computed as tenure * monthly_charges.",
        examples=[358.00]
    )
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Contract duration terms.",
        examples=["Month-to-month"]
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(
        ...,
        description="Customer payment method.",
        examples=["Electronic check"]
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Customer internet connection category.",
        examples=["Fiber optic"]
    )
    support_tickets: int = Field(
        default=0,
        ge=0,
        description="Count of logged customer service support incidents.",
        examples=[4]
    )
    last_login_days_ago: float = Field(
        ...,
        ge=0,
        description="Days elapsed since the customer's last active session.",
        examples=[28.0]
    )
    usage_frequency: float = Field(
        ...,
        ge=0,
        description="Product usage sessions / transactions per month.",
        examples=[14.0]
    )

    @model_validator(mode="after")
    def populate_defaults(self):
        if not self.customer_id:
            self.customer_id = f"CUST-{uuid.uuid4().hex[:8].upper()}"
        if self.total_charges is None:
            self.total_charges = round(self.tenure * self.monthly_charges, 2)
        return self


class RiskFactor(BaseModel):
    """
    Individual risk or protective factor derived from SHAP attribution values.
    """
    feature: str = Field(..., description="Feature or transformed feature name.")
    shap_value: float = Field(..., description="Impact on model log-odds / output.")
    transformed_value: float = Field(..., description="Standardized / encoded feature value.")


class InterventionRecommendation(BaseModel):
    """
    Recommended retention action based on business rules and SHAP risk drivers.
    """
    action_type: str = Field(..., description="Intervention action category.")
    priority: Literal["URGENT", "HIGH", "MEDIUM", "LOW"] = Field(..., description="Urgency priority.")
    reason: str = Field(..., description="Business rationale explaining the recommendation.")
    recommended_channel: str = Field(..., description="Suggested execution touchpoint.")
    suggested_incentive: Optional[str] = Field(None, description="Optional incentive, discount, or offer.")


class PredictionResponse(BaseModel):
    """
    Response schema returning churn inference, risk tier, SHAP explanations, and recommended actions.
    """
    customer_id: str
    churn_probability: float = Field(..., description="Predicted probability of churn between 0.0 and 1.0.")
    predicted_churn: int = Field(..., description="Binary classification (1 for Churn Risk, 0 for Retained).")
    risk_tier: str = Field(..., description="Risk category: CRITICAL RISK, ELEVATED RISK, or LOW RISK (LOYAL).")
    top_risk_factors: List[RiskFactor] = Field(..., description="Top contributing factors increasing churn probability.")
    top_protective_factors: List[RiskFactor] = Field(..., description="Top factors protecting customer against churn.")
    recommended_interventions: List[InterventionRecommendation] = Field(..., description="Targeted retention interventions.")
    persisted: bool = Field(True, description="Indicates if customer score and features were saved in SQLite.")


class InterventionCreate(BaseModel):
    """
    Payload for logging an intervention for a customer.
    """
    customer_id: str = Field(..., description="Target customer identifier.")
    action_type: str = Field(
        ...,
        description="Type of intervention applied or recommended.",
        examples=["Proactive Support Outreach", "Contract Upgrade Incentive"]
    )
    status: Literal["Pending", "Sent", "Responded", "Resolved"] = Field(
        default="Pending",
        description="Current lifecycle status of the intervention."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional operational or CSR notes.",
        examples=["Customer offered 20% discount on 1-year contract renewal."]
    )


class InterventionStatusUpdate(BaseModel):
    """
    Payload for updating an existing intervention status.
    """
    status: Literal["Pending", "Sent", "Responded", "Resolved"] = Field(
        ...,
        description="Updated lifecycle status."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional progress notes."
    )


class InterventionResponse(BaseModel):
    """
    Response schema for logged intervention records.
    """
    id: int
    customer_id: str
    action_type: str
    status: str
    notes: Optional[str] = None
    created_at: str
    updated_at: str


class CustomerListResponse(BaseModel):
    """
    Paginated response for customer records and stored churn scores.
    """
    total_count: int
    limit: int
    offset: int
    customers: List[Dict[str, Any]]


class LoginRequest(BaseModel):
    """
    Login request schema for user email authentication.
    """
    email: str = Field(..., description="Corporate user email address", examples=["analyst@churniq.ai"])
    password: Optional[str] = Field(default="", description="User account password")
    remember_me: Optional[bool] = Field(default=False, description="Persist session flag")


class UserProfile(BaseModel):
    """
    Authenticated user profile information.
    """
    email: str
    name: str
    role: str
    avatar_initials: str


class LoginResponse(BaseModel):
    """
    Successful authentication response with token and profile.
    """
    access_token: str
    token_type: str = "bearer"
    user: UserProfile
    message: str = "Authentication successful"

