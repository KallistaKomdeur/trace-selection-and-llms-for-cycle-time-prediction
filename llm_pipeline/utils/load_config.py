import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"

def load_config():
    """Loads settings from the configuration file.

    PPM_SETTINGS points at an alternative file, so two campaigns can run at once
    without sharing mutable state: fill_prompt re-reads this on every prompt, and a
    run whose selection_mode changed underneath it would silently switch fixed sets.
    """
    path = Path(os.environ.get("PPM_SETTINGS") or CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
