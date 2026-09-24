"""Put the repo root on the import path so pytest works from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
