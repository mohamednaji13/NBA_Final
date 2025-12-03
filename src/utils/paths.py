from .. import config

ROOT_DIR = config.ROOT_DIR
DATA_DIR = config.DATA_DIR
RAW_DIR = config.RAW_DIR
PROCESSED_DIR = config.PROCESSED_DIR
MODELS_DIR = config.MODELS_DIR
LOG_DIR = config.LOG_DIR

SCHEDULE_CSV = RAW_DIR / "games_schedule_5_seasons.csv"
BOXSCORES_CSV = RAW_DIR / "boxscores_5_seasons_v3.csv"
BOXSCORES_PARTIAL_CSV = RAW_DIR / "boxscores_partial_v3.csv"
DATASET_CSV = PROCESSED_DIR / "games_dataset_5_seasons.csv"
MODEL_FILE = MODELS_DIR / "home_win_classifier.pkl"
LOG_FILE = LOG_DIR / "nba_predictor.log"
