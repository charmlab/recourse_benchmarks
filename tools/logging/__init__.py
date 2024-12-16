import logging.config
import os
import pathlib

import yaml

lib_path = pathlib.Path(__file__).parent.resolve()
with open(os.path.join(lib_path, "logging.yaml"), "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

log = logging.getLogger(__name__)


def get_logger(logger: str):
    return logging.getLogger(logger)
