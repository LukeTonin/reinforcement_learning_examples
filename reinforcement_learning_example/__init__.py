from os.path import dirname, abspath

ROOT_DIR = dirname(abspath(__file__))

from reinforcement_learning_example.config.logging import configure_logger

configure_logger()