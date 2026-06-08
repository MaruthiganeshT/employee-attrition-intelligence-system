import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings

warnings.filterwarnings("ignore")


def get_explainer(model, X_train):
    """
    Initialize the appropriate SHAP explainer based on model type.
    """

    if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(
            model.predict_proba,
            background
        )

    return explainer


def save_top_features(shap_values, X_train):

    importance_values = np.abs(shap_values)
    
    print("TYPE:", type(shap_values))
    print("SHAPE:", np.array(shap_values).shape)
    print("IMPORTANCE SHAPE:", importance_values.shape)

    if len(importance_values.shape) == 3:
        importance_values = importance_values[:, :, 1]

    importance_values = importance_values.mean(axis=0)

    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance_values
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "eda_reports/top_features.csv",
        index=False
    )

    print(
        "Top features saved to "
        "eda_reports/top_features.csv"
    )

def generate_global_explanation(
    model,
    X_train,
    output_path="eda_reports/shap_summary.png"
):
    """
    Generate SHAP summary plot and feature importance plot.
    """

    explainer = get_explainer(model, X_train)

    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    os.makedirs("eda_reports", exist_ok=True)

    # Summary Plot
    plt.figure()

    shap.summary_plot(
        sv,
        X_train,
        show=False
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(
        f"SHAP summary plot saved to {output_path}"
    )

    # Feature Importance Plot
    plt.figure()

    shap.summary_plot(
        sv,
        X_train,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "eda_reports/shap_feature_importance.png"
    )

    plt.close()

    print(
        "Feature importance plot saved to "
        "eda_reports/shap_feature_importance.png"
    )

    # Save top features
    save_top_features(
        sv,
        X_train
    )

    return explainer


def generate_local_explanation(
    explainer,
    instance,
    feature_names
):
    """
    Generate SHAP values for a single prediction.
    """

    shap_values = explainer.shap_values(instance)

    expected_value = explainer.expected_value

    if isinstance(shap_values, list):

        sv = shap_values[1][0]

        ev = (
            expected_value[1]
            if isinstance(
                expected_value,
                (list, tuple, np.ndarray)
            )
            else expected_value
        )

    else:

        sv = shap_values[0]
        if len(np.array(sv).shape) == 2:
            sv = sv[:, 1]
        if isinstance(expected_value, np.ndarray):
            ev = expected_value[1]
        else:
            ev = expected_value

    return sv, ev


if __name__ == "__main__":

    print(
        "Generating Global SHAP Explainability..."
    )

    try:

        X_train = pd.read_csv(
            "data/processed/X_train.csv"
        )

        model = joblib.load(
            "models/best_model.joblib"
        )

        generate_global_explanation(
            model,
            X_train
        )

        print(
            "\nSHAP explainability completed "
            "successfully."
        )

    except FileNotFoundError:

        print(
            "Required files not found.\n"
            "Run preprocessing.py and "
            "model_trainer.py first."
        )

    except Exception as e:

        print(
            f"Unexpected error: {e}"
        )