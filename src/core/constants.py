import os
from enum import Enum
from pathlib import Path

"""
File to store constants
"""

class DirectoryPath(Enum):
    """
    directories path
    """
    CHROMA_PERSIST_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"/ "vectors"
    CONFIG_FILE = Path(__file__).resolve().parent.parent.parent / "configs"/ "config.yaml"
    STREAMLIT_APP = Path(__file__).resolve().parent.parent / "app"/ "main.py"