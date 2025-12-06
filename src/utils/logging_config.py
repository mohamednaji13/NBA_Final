# src/utils/logging_config.py

import logging
from pathlib import Path

def setup_logging(name: str):
    """
    Configure consistent logging for all pipeline modules.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when run repeatedly
    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)

    logger.addHandler(console)
    return logger
