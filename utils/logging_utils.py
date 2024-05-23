import logging
import sys

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
DEFAULT_LOG_DATE = "%m/%d/%Y %H:%M:%S"
DEFAULT_LOG_FORMATTER = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_LOG_DATE)

logging.basicConfig(
    format=DEFAULT_LOG_FORMAT,
    datefmt=DEFAULT_LOG_DATE,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# from logging.handlers import RotatingFileHandler
# LOG_FILENAME = os.path.join(BASE_DIR, 'gec_sys.log')
# LOG_HANDLER = RotatingFileHandler(
#     filename=LOG_FILENAME,
#     mode="a",
#     maxBytes=50 * 1024 * 1024,
#     backupCount=5,
#     encoding="utf-8",
# )

# LOG_HANDLER = logging.StreamHandler()
# LOG_HANDLER.setFormatter(LOG_FORMATTER)


def get_logger(name: str, level=DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Initialises and returns named Django logger instance."""
    named_logger = logging.getLogger(name=name)
    named_logger.setLevel(level)

    # named_logger.addHandler(LOG_HANDLER)
    return named_logger
