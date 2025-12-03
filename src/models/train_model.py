import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
    if not paths.DATASET_CSV.exists():
        logger.error(f"Dataset CSV not found at {paths.DATASET_CSV}. Run build_dataset first.")
        return

    df = pd.read_csv(paths.DATASET_CSV)
    if "home_team_won" not in df.columns:
        logger.error("Target column 'home_team_won' not found.")
        return

    numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    logger.info(f"Using {len(feature_cols)} numeric features for training.")

    X = df[feature_cols]
    y = df["home_team_won"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("Training RandomForestClassifier...")
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test ROC-AUC:  {auc:.4f}")

    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, paths.MODEL_FILE)
    logger.info(f"Saved model to {paths.MODEL_FILE}")

if __name__ == "__main__":
    main()
