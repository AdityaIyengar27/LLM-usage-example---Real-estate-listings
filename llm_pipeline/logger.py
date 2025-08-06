import logging


def setup_logger(name: str = "llm_pipeline", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    :param name: str, the name of the logger
    :param level: int, the logging level (default is logging.INFO)
    :return: logging.Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
