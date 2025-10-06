import os
from appdirs import user_log_dir
import logging
from logging.handlers import RotatingFileHandler
from logging import StreamHandler

APP_NAME = "SAVANT"

LOG_DIR = user_log_dir(APP_NAME)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "savant_app.log")

def setup_logger() -> None:
    """Set up logging configuration."""

    logging.basicConfig(filename='savant_app.log', level=logging.INFO) # Configure logging to file, log from INFO to CRITICAL 

    logger = logging.getLogger()

    # File handler
    file_handler = RotatingFileHandler(LOG_FILE , maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(file_formatter)
    logger.addHandler(stream_handler)