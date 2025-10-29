"""
Logging utilities for consistent logging configuration across the data processing pipeline.

This module provides centralized logging setup to eliminate duplicate logging
configuration code across multiple modules.
"""

import logging
from typing import Optional


def configure_logger(name: str,
                    level: int = logging.INFO,
                    suppress_external: bool = True,
                    format_string: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger with consistent settings.

    This function provides a centralized way to configure loggers across the
    data processing pipeline, ensuring consistent behavior and reducing code duplication.

    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Logging level (default: logging.INFO)
        suppress_external: If True, suppress verbose output from external libraries
        format_string: Custom format string for log messages (optional)

    Returns:
        Configured logging.Logger instance

    Examples:
        >>> logger = configure_logger(__name__)
        >>> logger.info("Processing started")

        >>> logger = configure_logger(__name__, level=logging.DEBUG, suppress_external=False)
        >>> logger.debug("Detailed debug information")
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Configure basic logging if not already configured
    if not logging.getLogger().handlers:
        if format_string:
            logging.basicConfig(level=level, format=format_string)
        else:
            logging.basicConfig(level=level)

    # Suppress verbose output from external libraries
    if suppress_external:
        # Suppress urllib3 connection pool logging
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

        # Suppress pdfminer logging (used in children's books scraper)
        logging.getLogger('pdfminer').setLevel(logging.ERROR)

        # Suppress other common noisy libraries
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('charset_normalizer').setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with default configuration.

    Convenience function for getting a logger with standard settings.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured logging.Logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Using default configuration")
    """
    return configure_logger(name)


def set_log_level(logger: logging.Logger, level: int) -> None:
    """Change the logging level of an existing logger.

    Args:
        logger: Logger instance to modify
        level: New logging level (e.g., logging.DEBUG, logging.INFO)

    Examples:
        >>> logger = get_logger(__name__)
        >>> set_log_level(logger, logging.DEBUG)
    """
    logger.setLevel(level)
    # Also update root logger if needed
    if logging.getLogger().level > level:
        logging.getLogger().setLevel(level)
