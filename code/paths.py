import os
from sqlite3 import DatabaseError

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_DIR = os.path.join(ROOT_DIR, "code")

DATA_DIR = os.path.join(ROOT_DIR, "data")

OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

CONFIG_FILE_PATH = os.path.join(CODE_DIR, "config.yaml")

DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
