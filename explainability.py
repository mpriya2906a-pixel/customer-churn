import os
import sys
import types
import joblib
import numpy as np
import pandas as pd

# Fallback shim for environments where numba compiled DLLs are blocked
if "numba" not in sys.modules:
    try:
        import numba  # noqa: F401
    except Exception:
        class _MockMod(types.ModuleType):
            def __getattr__(self, name):
                m = _MockMod(f"{self.__name__}.{name}")
                m.__path__ = []
                m.njit = lambda *a, **kw: (lambda f: f)
                m.jit = lambda *a, **kw: (lambda f: f)
                return m

        _m = _MockMod("numba")
        _m.__path__ = []
        _m.njit = lambda *a, **kw: (lambda f: f)
        _m.jit = lambda *a, **kw: (lambda f: f)
        sys.modules["numba"] = _m
        sys.modules["numba.typed"] = _MockMod("numba.typed")
        sys.modules["numba.core"] = _MockMod("numba.core")
        sys.modules["numba.core.types"] = _MockMod("numba.core.types")
        sys.modules["numba.experimental"] = _MockMod("numba.experimental")

import shap
from typing import Dict, Any, List, Optional


class ChurnExplainer:
    """
    SHAP-based explainability engine for Customer Churn models.
    Supports local (per-customer) explanations and global feature importance.
    """
    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.model_type = type(model).__name__
        self.explainer = self._build_explainer(model, background_data)

    def __getstate__(self):
        # Exclude internal SHAP explainer instance containing unpicklable lambdas
        return {
            "model": self.model,
            "feature_names": self.feature_names,
            "background_data": self.background_data,
            "model_type": self.model_type
        }

    def __setstate__(self, state):
        self.model = state["model"]
        self.feature_names = state["feature_names"]
        self.background_data = state.get("background_data")
        self.model_type = state.get("model_type", type(self.model).__name__)
        self.explainer = self._build_explainer(self.model, self.background_data)
        
    def _build_explainer(self, model: Any, background_data: Optional[np.ndarray]):
        """
        Builds the appropriate SHAP explainer based on model architecture.
        """
        # Tree-based models (XGBoost, RandomForest)
        if hasattr(model, "estimators_") or "XGB" in self.model_type or "Forest" in self.model_type:
            print(f"[ChurnExplainer] Initializing TreeExplainer for {self.model_type}...")
            return shap.TreeExplainer(model)
        else:
            print(f"[ChurnExplainer] Initializing LinearExplainer / Explainer for {self.model_type}...")
            if background_data is not None:
                sample_bg = background_data[:min(len(background_data), 100)]
                return shap.LinearExplainer(model, sample_bg)
            return shap.LinearExplainer(model, np.zeros((1, len(self.feature_names))))

    def _extract_shap_array(self, shap_output: Any) -> np.ndarray:
        """
        Normalizes SHAP output to a 2D numpy array (samples x features) for the positive class (Churn).
        """
        if hasattr(shap_output, "values"):
            vals = shap_output.values
        else:
            vals = shap_output

        vals = np.array(vals)

        # If shape is (n_samples, n_features, 2) -> binary classification, take class 1
        if vals.ndim == 3 and vals.shape[2] == 2:
            return vals[:, :, 1]
        # If list of arrays (one per class)
        elif isinstance(shap_output, list) and len(shap_output) == 2:
            return np.array(shap_output[1])
        # If shape is (n_samples, n_features)
        elif vals.ndim == 2:
            return vals
        elif vals.ndim == 1:
            return vals.reshape(1, -1)
        else:
            return vals

    def explain_instance(
        self,
        customer_processed_row: np.ndarray,
        customer_raw_data: Optional[Dict[str, Any]] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Computes local SHAP explanation for a single customer.
        Returns top risk factors (pushing toward churn) and protective factors (pushing toward staying).
        """
        if customer_processed_row.ndim == 1:
            row_2d = customer_processed_row.reshape(1, -1)
        else:
            row_2d = customer_processed_row

        # Model prediction & probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(row_2d)[0]
            churn_prob = float(probs[1])
        else:
            churn_prob = float(self.model.predict(row_2d)[0])
            
        churn_pred = int(churn_prob >= 0.5)

        # Compute SHAP values
        raw_shap = self.explainer(row_2d)
        shap_vals = self._extract_shap_array(raw_shap)[0]

        # Combine feature names, feature values, and SHAP values
        feature_impacts = []
        for i, fname in enumerate(self.feature_names):
            s_val = float(shap_vals[i])
            f_val = float(row_2d[0, i])
            feature_impacts.append({
                "feature": fname,
                "shap_value": round(s_val, 4),
                "transformed_value": round(f_val, 4)
            })

        # Sort by SHAP value
        # Positive SHAP value -> increases churn probability (risk factor)
        # Negative SHAP value -> decreases churn probability (retention / protective factor)
        risk_factors = sorted(
            [f for f in feature_impacts if f["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True
        )[:top_k]

        protective_factors = sorted(
            [f for f in feature_impacts if f["shap_value"] < 0],
            key=lambda x: x["shap_value"]
        )[:top_k]

        # Risk tier assessment
        if churn_prob >= 0.70:
            risk_tier = "CRITICAL RISK"
        elif churn_prob >= 0.40:
            risk_tier = "ELEVATED RISK"
        else:
            risk_tier = "LOW RISK (LOYAL)"

        return {
            "customer_id": customer_raw_data.get("customer_id", "Unknown") if customer_raw_data else "N/A",
            "churn_probability": round(churn_prob, 4),
            "predicted_churn": churn_pred,
            "risk_tier": risk_tier,
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
            "raw_customer_attributes": customer_raw_data or {}
        }

    def get_global_importance(self, X_sample: np.ndarray) -> pd.DataFrame:
        """
        Computes mean absolute SHAP value across a representative sample.
        """
        raw_shap = self.explainer(X_sample)
        shap_vals = self._extract_shap_array(raw_shap)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return importance_df


def save_explainer(explainer: ChurnExplainer, filepath: str):
    """
    Saves the explainer object to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(explainer, filepath)
    print(f"[ChurnExplainer] Saved explainer to {filepath}")


def load_explainer(filepath: str) -> ChurnExplainer:
    """
    Loads the explainer object from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Explainer artifact not found at {filepath}")
    return joblib.load(filepath)
