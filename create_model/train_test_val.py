from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


TARGET_COL = "POINT_DIFF"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models_store"


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    return pd.read_csv(path)


def features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe.")

    y = df[TARGET_COL]
    drop_cols = [c for c in df.columns if "PLUS_MINUS" in c.upper()]
    X = df.drop(columns=drop_cols + [TARGET_COL])

    # Keep numeric/bool features only and fill any gaps.
    num_cols = X.select_dtypes(include=["number", "bool"]).columns
    X = X[num_cols].fillna(0)
    return X, y


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    # Elastic Net search
    enet_grid = [{"alpha": a, "l1_ratio": l1} for a in [0.0005, 0.001, 0.005] for l1 in [0.1, 0.5, 0.9]]
    best_enet = None
    best_enet_metrics = None
    best_enet_params = None
    for params in enet_grid:
        model = ElasticNet(max_iter=5000, random_state=42, **params)
        model.fit(X_train, y_train)
        metrics = evaluate(y_val, model.predict(X_val))
        if best_enet_metrics is None or metrics["mse"] < best_enet_metrics["mse"]:
            best_enet = model
            best_enet_metrics = metrics
            best_enet_params = params
    results["elastic_net"] = {"model": best_enet, "metrics": best_enet_metrics, "params": best_enet_params}
    print(f"elastic_net best params {best_enet_params} metrics {best_enet_metrics}")

    # Random Forest search
    rf_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
        for n in [200, 400]
        for d in [None, 20]
        for m in [1, 2]
    ]
    best_rf = None
    best_rf_metrics = None
    best_rf_params = None
    for params in rf_grid:
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        metrics = evaluate(y_val, model.predict(X_val))
        if best_rf_metrics is None or metrics["mse"] < best_rf_metrics["mse"]:
            best_rf = model
            best_rf_metrics = metrics
            best_rf_params = params
    results["random_forest"] = {"model": best_rf, "metrics": best_rf_metrics, "params": best_rf_params}
    print(f"random_forest best params {best_rf_params} metrics {best_rf_metrics}")

    # XGBoost search (if available)
    if HAS_XGB:
        xgb_grid = [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.8, "colsample_bytree": 0.8}
            for n in [300, 500]
            for d in [4, 6]
            for lr in [0.03, 0.07]
        ]
        best_xgb = None
        best_xgb_metrics = None
        best_xgb_params = None
        for params in xgb_grid:
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                **params,
            )
            model.fit(X_train, y_train)
            metrics = evaluate(y_val, model.predict(X_val))
            if best_xgb_metrics is None or metrics["mse"] < best_xgb_metrics["mse"]:
                best_xgb = model
                best_xgb_metrics = metrics
                best_xgb_params = params
        results["xgboost"] = {"model": best_xgb, "metrics": best_xgb_metrics, "params": best_xgb_params}
        print(f"xgboost best params {best_xgb_params} metrics {best_xgb_metrics}")

    return results


def main() -> None:
    print("Loading train/val/test splits...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train = features_and_target(train_df)
    X_val, y_val = features_and_target(val_df)
    X_test, y_test = features_and_target(test_df)

    results = train_models(X_train, y_train, X_val, y_val)

    # Select best model by lowest validation MSE.
    best_name = min(results, key=lambda n: results[n]["metrics"]["mse"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]
    print(f"\nBest model on validation: {best_name} with metrics {best_metrics}")

    # Refit best model on train + val, then evaluate on test.
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    print(f"Refitting best model ({best_name}) on train+val...")
    best_model.fit(X_trainval, y_trainval)

    test_pred = best_model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)
    print(f"{best_name} test metrics: {test_metrics}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{best_name}_best_model.joblib"
    dump(best_model, model_path)
    print(f"Saved best model to {model_path.resolve()}")


if __name__ == "__main__":
    main()
