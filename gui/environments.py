"""
WiLoR lives in its own Python environment (torch + CUDA, non-commercial
licenses), separate from the light one the app normally runs in. This finds
it, so the app can restart itself there instead of asking the user to.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def wilor_installed_here() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("wilor_mini") is not None


def wilor_launcher() -> Optional[Tuple[Path, Path]]:
    """
    (python, app script) of the WiLoR environment, or None if it isn't
    installed or this already is it. From source that is .venv-wilor next to
    app.py; in the release package, depth\\.venv made by install-depth.bat.
    """
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent.parent
        python, script = root / "depth" / ".venv" / "Scripts" / "python.exe", root / "depth" / "src" / "app.py"
    else:
        repo = Path(__file__).resolve().parent.parent
        python, script = repo / ".venv-wilor" / "Scripts" / "python.exe", repo / "app.py"
    if not (python.exists() and script.exists()):
        return None
    if Path(sys.executable).resolve() == python.resolve():
        return None
    return python, script


def relaunch(python: Path, script: Path, config_path: str):
    """Start the app again in another environment, with the same working folder and config."""
    subprocess.Popen([str(python), str(script), os.path.abspath(config_path)], cwd=os.getcwd())
