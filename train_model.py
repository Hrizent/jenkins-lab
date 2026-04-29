from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "breast_cancer.csv"
BEST_MODEL_PATH_FILE = Path("best_model.txt")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data():
    if RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        cancer = load_breast_cancer(as_frame=True)
        df = cancer.frame.copy()

    x = df.drop("target", axis=1)
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test


def main():
    x_train_scaled, x_test_scaled, y_train, y_test = load_data()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Breast Cancer Prediction")

    models = {
        "logistic_regression": LogisticRegression(
            C=0.1,
            solver="liblinear",
            random_state=42,
            max_iter=1000,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        ),
        "svc": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=42,
        ),
    }

    results = []

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            rmse, mae, r2 = eval_metrics(y_test, y_pred)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            signature = mlflow.models.infer_signature(
                x_train_scaled,
                model.predict(x_train_scaled),
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
            )

            run_id = mlflow.active_run().info.run_id
            results.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )

            print(
                f"Model: {model_name}, Run ID: {run_id}, "
                f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(
        ["accuracy", "f1"],
        ascending=False,
    )

    best_run = results_df.iloc[0]
    model_uri = f"runs:/{best_run['run_id']}/model"

    BEST_MODEL_PATH_FILE.write_text(model_uri, encoding="utf-8")

    print("\nBest model selected:")
    print(f"Model: {best_run['model_name']}")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Model URI: {model_uri}")
    print(f"Saved to: {BEST_MODEL_PATH_FILE}")


if __name__ == "__main__":
    main()

