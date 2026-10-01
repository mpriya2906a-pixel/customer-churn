"""
Rule-Based Intervention Recommender.
Maps customer risk drivers, behavioral attributes, and SHAP factors to actionable retention interventions.
"""

from typing import Dict, Any, List, Optional


class InterventionRecommender:
    """
    Intelligent rule-based engine that converts churn drivers into prioritized,
    actionable retention interventions.
    """

    @classmethod
    def recommend(
        cls,
        customer_data: Dict[str, Any],
        top_risk_factors: Optional[List[Dict[str, Any]]] = None,
        churn_probability: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Analyzes customer attributes and SHAP risk drivers to generate prioritized recommendations.
        """
        top_risk_factors = top_risk_factors or []
        risk_features = {f.get("feature", "") for f in top_risk_factors}

        support_tickets = int(customer_data.get("support_tickets", 0))
        contract_type = str(customer_data.get("contract_type", ""))
        last_login_days_ago = float(customer_data.get("last_login_days_ago", 0))
        usage_frequency = float(customer_data.get("usage_frequency", 0))
        monthly_charges = float(customer_data.get("monthly_charges", 0.0))
        payment_method = str(customer_data.get("payment_method", ""))
        tenure = float(customer_data.get("tenure", 0))
        internet_service = str(customer_data.get("internet_service", ""))

        recommendations: List[Dict[str, Any]] = []

        # 1. High Support Tickets Rule
        if support_tickets >= 3 or "support_tickets" in risk_features:
            priority = "URGENT" if (support_tickets >= 4 or churn_probability >= 0.75) else "HIGH"
            recommendations.append({
                "action_type": "Proactive Support Outreach",
                "priority": priority,
                "reason": f"Customer logged {support_tickets} support tickets; service frustration is a primary churn risk driver.",
                "recommended_channel": "Dedicated Phone Call / Senior CSM Escalation",
                "suggested_incentive": "Expedited technical resolution + 1-month service credit"
            })

        # 2. Month-to-Month Contract Rule
        if contract_type == "Month-to-month" or any("contract_type" in rf for rf in risk_features):
            priority = "HIGH" if churn_probability >= 0.60 else "MEDIUM"
            recommendations.append({
                "action_type": "Contract Upgrade Incentive",
                "priority": priority,
                "reason": "Month-to-month contract offers minimal switching friction; long-term commitment needed.",
                "recommended_channel": "Targeted Email & In-App Banner",
                "suggested_incentive": "15% discount or 2 months free upon upgrading to a 1-year or 2-year plan"
            })

        # 3. Inactivity & Low Usage Rule
        if usage_frequency < 25.0 or last_login_days_ago >= 20 or "last_login_days_ago" in risk_features or "usage_frequency" in risk_features:
            priority = "HIGH" if last_login_days_ago >= 35 else "MEDIUM"
            recommendations.append({
                "action_type": "Re-engagement & Adoption Campaign",
                "priority": priority,
                "reason": f"Dormant activity observed: last active {last_login_days_ago:.0f} days ago, usage frequency {usage_frequency:.1f}/month.",
                "recommended_channel": "Automated Personalized Email Workflow",
                "suggested_incentive": "Complimentary feature walkthrough session + 14-day premium add-on access"
            })

        # 4. High Monthly Charges / Bill Shock Rule
        if monthly_charges >= 75.0 or "monthly_charges" in risk_features:
            recommendations.append({
                "action_type": "Plan Optimization & Loyalty Review",
                "priority": "MEDIUM",
                "reason": f"Monthly billing of ${monthly_charges:.2f} is higher than average, causing cost sensitivity.",
                "recommended_channel": "Account Manager Outreach / Billing Consultation",
                "suggested_incentive": "Custom bundle optimization saving $15-25/month without feature loss"
            })

        # 5. Payment Method Friction Rule
        if payment_method == "Electronic check" or "payment_method_Electronic check" in risk_features:
            recommendations.append({
                "action_type": "Auto-Pay Card Enrollment Bonus",
                "priority": "LOW",
                "reason": "Electronic check payments correlate with higher involuntary and voluntary churn rates.",
                "recommended_channel": "Payment Portal Notification / SMS Prompt",
                "suggested_incentive": "$10 one-time account credit for enabling automatic credit card / ACH billing"
            })

        # 6. Early-Tenure Flight Risk Rule
        if tenure <= 6 and churn_probability >= 0.50:
            recommendations.append({
                "action_type": "New Customer Concierge Check-in",
                "priority": "HIGH",
                "reason": f"Customer is in the critical first {int(tenure)} months of relationship where initial friction leads to rapid churn.",
                "recommended_channel": "Customer Success Manager Welcome Call",
                "suggested_incentive": "Personalized 1-on-1 setup session"
            })

        # Fallback / Low Risk Customers
        if not recommendations:
            if churn_probability < 0.35:
                recommendations.append({
                    "action_type": "Loyalty Appreciation & Advocacy",
                    "priority": "LOW",
                    "reason": "Customer exhibits healthy engagement and low churn propensity.",
                    "recommended_channel": "Quarterly Newsletter / In-App Reward",
                    "suggested_incentive": "Refer-a-friend credit or loyalty anniversary badge"
                })
            else:
                recommendations.append({
                    "action_type": "General Account Health Review",
                    "priority": "MEDIUM",
                    "reason": f"Elevated churn probability ({churn_probability:.1%}) without a single dominant risk factor.",
                    "recommended_channel": "Customer Success Survey & Check-in",
                    "suggested_incentive": "Satisfaction survey with guaranteed response follow-up"
                })

        # Sort recommendations by urgency: URGENT > HIGH > MEDIUM > LOW
        priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "MEDIUM"), 99))

        return recommendations
