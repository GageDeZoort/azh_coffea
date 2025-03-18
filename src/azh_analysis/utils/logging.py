from __future__ import annotations

import logging


def init_logging(verbose=False):
    log_format = "%(asctime)s %(levelname)s %(message)s"
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info("Initializing processor logger.")
