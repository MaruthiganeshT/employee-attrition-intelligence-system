import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_processed_data():
    """Load preprocessed train and test data."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        "data/processed/y_test.csv"
    ).squeeze("columns")

    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train multiple models and evaluate them."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    best_model = None
    best_model_name = ""
    best_f1 = 0

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Cross Validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        cv_f1 = cv_scores.mean()

        # Train on full training set
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)

        prec = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        rec = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_proba
        )

        results.append({
            "Model": name,
            "CV F1": round(cv_f1, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    print(
        f"\nBest Model: {best_model_name}"
    )

    print(
        f"Best F1 Score: {best_f1:.4f}"
    )

    return (
        best_model,
        best_model_name,
        results_df
    )


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = (
        load_processed_data()
    )

    (
        best_model,
        best_model_name,
        results_df
    ) = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("eda_reports", exist_ok=True)

    joblib.dump(
        best_model,
        "models/best_model.joblib"
    )

    results_df.to_csv(
        "eda_reports/model_metrics.csv",
        index=False
    )

    print(
        "\nBest model and metrics saved successfully."
    )