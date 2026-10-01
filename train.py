import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)

from src.pipeline import (
    build_preprocessor,
    prepare_features_and_target,
    get_transformed_feature_names,
    save_pipeline,
    create_full_pipeline
)
from src.data_generator import load_or_generate_dataset


def get_models(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """
    Initializes the three candidate classification models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates a trained model and returns comprehensive performance metrics.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "roc_auc": float(roc_auc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "cm_raw": cm,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def print_evaluation_summary(results: Dict[str, Dict[str, Any]]):
    """
    Prints a nicely formatted comparison table and confusion matrices.
    """
    print("\n" + "=" * 80)
    print("                      MODEL PERFORMANCE EVALUATION")
    print("=" * 80)
    print(f"{'Model':<22} | {'ROC-AUC':<9} | {'F1-Score':<9} | {'Precision':<10} | {'Recall':<8} | {'Accuracy':<8}")
    print("-" * 80)
    for name, metrics in results.items():
        print(
            f"{name:<22} | "
            f"{metrics['roc_auc']:<9.4f} | "
            f"{metrics['f1_score']:<9.4f} | "
            f"{metrics['precision']:<10.4f} | "
            f"{metrics['recall']:<8.4f} | "
            f"{metrics['accuracy']:<8.4f}"
        )
    print("-" * 80)

    for name, metrics in results.items():
        cm = metrics['cm_raw']
        print(f"\n[Confusion Matrix: {name}]")
        print(f"                 Predicted 0 (Stay)    Predicted 1 (Churn)")
        print(f"  Actual 0 (Stay)    {cm[0, 0]:>6} (TN)              {cm[0, 1]:>6} (FP)")
        print(f"  Actual 1 (Churn)   {cm[1, 0]:>6} (FN)              {cm[1, 1]:>6} (TP)")


def train_and_evaluate(
    data_path: str = None,
    artifacts_dir: str = "models",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Any, Any, Dict[str, Any], np.ndarray, np.ndarray, list]:
    """
    Complete training pipeline:
    1. Loads or generates dataset
    2. Fits preprocessing pipeline
    3. Trains Logistic Regression, Random Forest, XGBoost
    4. Evaluates and selects the best model
    5. Saves artifacts with joblib
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Load Data
    df = load_or_generate_dataset() if data_path is None else pd.read_csv(data_path)
    X, y = prepare_features_and_target(df)
    
    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"\n[Split] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    print(f"[Split] Class distribution - Train: {y_train.mean():.1%} churn | Test: {y_test.mean():.1%} churn")

    # 3. Fit Preprocessing Pipeline
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_transformed_feature_names(preprocessor)
    print(f"[Preprocessor] Transformed {len(feature_names)} features: {feature_names}")

    # Compute scale_pos_weight for XGBoost to handle class balance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    # 4. Train Models
    models = get_models(scale_pos_weight=scale_pos_weight)
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n[Training] Fitting {name}...")
        model.fit(X_train_proc, y_train)
        eval_metrics = evaluate_model(model, X_test_proc, y_test.values)
        results[name] = eval_metrics
        trained_models[name] = model

    # 5. Summary & Model Selection
    print_evaluation_summary(results)
    
    # Best model chosen primarily by ROC-AUC
    best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print("\n" + "#" * 80)
    print(f" WINNER: >>> {best_model_name.upper()} <<<")
    print(f" ROC-AUC: {best_metrics['roc_auc']:.4f} | F1-Score: {best_metrics['f1_score']:.4f}")
    print("#" * 80)

    # 6. Save Artifacts (joblib)
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    model_path = os.path.join(artifacts_dir, "best_model.joblib")
    full_pipeline_path = os.path.join(artifacts_dir, "full_pipeline.joblib")
    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")

    save_pipeline(preprocessor, preprocessor_path)
    save_pipeline(best_model, model_path)
    
    # Create and save unified scikit-learn Pipeline
    full_pipe = create_full_pipeline(classifier=best_model, preprocessor=preprocessor)
    save_pipeline(full_pipe, full_pipeline_path)

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "metrics": {
            k: {m: v for m, v in metrics.items() if m not in ['cm_raw', 'y_pred', 'y_prob']}
            for k, metrics in results.items()
        },
        "artifacts": {
            "preprocessor": preprocessor_path,
            "best_model": model_path,
            "full_pipeline": full_pipeline_path
        }
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[Artifacts] Metadata written to {metadata_path}")

    return best_model, preprocessor, results, X_test_proc, y_test.values, feature_names


if __name__ == "__main__":
    train_and_evaluate()
