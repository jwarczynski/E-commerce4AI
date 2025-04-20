import argparse
import logging

_global_log_level = logging.DEBUG  # Default global log level


def set_global_log_level():
    """Set the global logging level based on command-line arguments."""
    global _global_log_level
    parser = argparse.ArgumentParser(description='Run the main script with configurable logging.')
    parser.add_argument(
        '--log-level',
        default='DEBUG',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    args = parser.parse_args()
    _global_log_level = getattr(logging, args.log_level.upper(), logging.DEBUG)


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with console output using the global log level."""
    logger = logging.getLogger(name)
    logger.setLevel(_global_log_level)  # Use the global log level
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
