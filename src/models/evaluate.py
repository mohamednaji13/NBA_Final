import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from ..utils.logging_config import setup_logging
from ..utils import paths
from .. import config

logger = setup_logging(__name__)

EXCLUDE_COLS = {
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "home_team_won",
}

def main():
    if not paths.MODEL_FILE.exists():
        logger.error(f"Model file not found at {paths.MODEL_FILE}. Train the model first.")
        return
    if not paths.DATASET_CSV.exists():
        logger.error(f"Dataset CSV not found at {paths.DATASET_CSV}.")
        return

    clf = joblib.load(paths.MODEL_FILE)
    df = pd.read_csv(paths.DATASET_CSV)
    if "home_team_won" not in df.columns:
        logger.error("Target column 'home_team_won' not found.")
        return

    numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    X = df[feature_cols]
    y = df["home_team_won"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"[EVAL] Test Accuracy: {acc:.4f}")
    logger.info(f"[EVAL] Test ROC-AUC:  {auc:.4f}")

if __name__ == "__main__":
    main()
