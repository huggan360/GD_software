# ---------------------
# READ THE TOOL SETTINGS
# ---------------------
from pathlib import Path

import yaml


# ---------------------
# Load service settings
# ---------------------
def read_settings(path):
    """Resolve paths next to the config file, regardless of the terminal folder."""
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text())
    for key in ("storage", "video_config"):
        settings[key] = str((path.parent / settings[key]).resolve())
    return settings


# ---------------------
# The fridge's current mode decides which page the tool shows
# ---------------------
def read_mode(settings):
    """acquisition, test or production; production when the mode file is missing."""
    path = settings.get("mode_file")
    if path and Path(path).is_file():
        return Path(path).read_text().strip()
    return "production"
