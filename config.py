import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
PARENT_DIR = BASE_DIR.parent
PYTORCH_NEAT_PATH = os.environ.get("PYTORCH_NEAT_PATH", BASE_DIR.parent / "PyTorch-NEAT")
