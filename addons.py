"""
Add-ons the app can install for the user: the SteamVR driver, and the optional
WiLoR depth environment. This module only finds, checks and copies; the
window that offers them is gui/addons_dialog.py.

Where things are depends on how the app runs:
- From source: the driver as built in "SteamVR Driver", WiLoR in .venv-wilor.
- Release package: everything ships in an "addons" folder (marked by
  package.json) inside the app, and WiLoR installs into depth\\.venv next to
  the app. The WiLoR environment runs our sources from addons\\depth\\src, so
  this has to work from there too.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from version import APP_VERSION

DRIVER_NAME = "HandTrackCamVR"          # "name" in driver.vrdrivermanifest
DRIVER_FOLDER = "handcameradriver"      # folder under SteamVR\drivers
DRIVER_DLL = "driver_HandTrackCamVR.dll"
DRIVER_VERSION_FILE = "handcam-version.json"
STEAMVR_PROCESSES = ("vrserver.exe", "vrmonitor.exe", "vrcompositor.exe")
# Bump when install-depth.ps1 installs something different, so older installs are offered an update
DEPTH_VERSION = 1
DEPTH_MARKER = "handcam-depth.json"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AddonError(Exception):
    """Something the user can fix, worded for them."""


# ----- where things are

@dataclass(frozen=True)
class Layout:
    packaged: bool
    app_dir: Path             # holds config.json; the exe in the release package
    driver_files: Tuple[Tuple[Path, str], ...]  # (bundled file, path inside the installed driver folder)
    depth_env: Path           # WiLoR's Python environment
    depth_installer: Path     # install-depth.ps1
    depth_requirements: Path
    depth_constraints: Path
    depth_app: Path           # app.py run inside the WiLoR environment

    @property
    def version(self) -> str:
        return APP_VERSION if self.packaged else f"{APP_VERSION} (from source)"

    @property
    def depth_python(self) -> Path:
        return self.depth_env / "Scripts" / "python.exe"


def _package_addons_dir() -> Optional[Path]:
    if getattr(sys, "frozen", False):
        candidate = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "addons"
        return candidate if (candidate / "package.json").exists() else None
    # Running from addons\depth\src inside the package (the WiLoR environment)
    here = Path(__file__).resolve().parent
    candidate = here.parent.parent
    if here.name == "src" and (candidate / "package.json").exists():
        return candidate
    return None


def _folder_files(root: Path) -> List[Tuple[Path, str]]:
    return [(p, p.relative_to(root).as_posix()) for p in sorted(root.rglob("*")) if p.is_file()]


def layout() -> Layout:
    addons_dir = _package_addons_dir()
    if addons_dir is not None:
        app_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else addons_dir.parent.parent
        driver_dir = addons_dir / "driver" / DRIVER_FOLDER
        depth = addons_dir / "depth"
        files = _folder_files(driver_dir) if (driver_dir / "bin" / "win64" / DRIVER_DLL).exists() else []
        return Layout(True, app_dir, tuple(files), app_dir / "depth" / ".venv", depth / "install-depth.ps1",
                      depth / "requirements.txt", depth / "constraints.txt", depth / "src" / "app.py")
    repo = Path(__file__).resolve().parent
    driver = repo / "SteamVR Driver"
    dll = driver / "bin" / "win64" / DRIVER_DLL
    files: List[Tuple[Path, str]] = []
    if dll.exists():
        files.append((driver / "manifest" / "driver.vrdrivermanifest", "driver.vrdrivermanifest"))
        files.append((dll, f"bin/win64/{DRIVER_DLL}"))
        files += [(p, f"resources/{rel}") for p, rel in _folder_files(driver / "resources")]
    installers = repo / "installers"
    return Layout(False, repo, tuple(files), repo / ".venv-wilor", installers / "install-depth.ps1",
                  repo / "requirements.txt", installers / "depth-constraints.txt", repo / "app.py")


# ----- SteamVR

def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _openvr_paths() -> dict:
    local = os.environ.get("LOCALAPPDATA")
    return (_read_json(Path(local) / "openvr" / "openvrpaths.vrpath") if local else None) or {}


def steam_dir() -> Optional[Path]:
    if sys.platform != "win32":
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            path = Path(winreg.QueryValueEx(key, "SteamPath")[0])
    except OSError:
        return None
    return path if path.exists() else None


def find_steamvr() -> Optional[Path]:
    """SteamVR's install folder: the runtime OpenVR uses, else the Steam libraries."""
    candidates = [Path(p) for p in _openvr_paths().get("runtime", [])]
    steam = steam_dir()
    if steam is not None:
        libraries = [steam]
        vdf = steam / "steamapps" / "libraryfolders.vdf"
        try:
            import re
            text = vdf.read_text(encoding="utf-8", errors="replace")
            libraries += [Path(m.replace("\\\\", "\\")) for m in re.findall(r'"path"\s+"([^"]+)"', text)]
        except OSError:
            pass
        candidates += [lib / "steamapps" / "common" / "SteamVR" for lib in libraries]
    for candidate in candidates:
        if (candidate / "bin").is_dir() and (candidate / "drivers").is_dir():
            return candidate
    return None


def steamvr_running() -> bool:
    if sys.platform != "win32":
        return False
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                             timeout=10, creationflags=NO_WINDOW).stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return False
    return any(f'"{name}"' in out for name in STEAMVR_PROCESSES)


def _steamvr_settings_path() -> Optional[Path]:
    steam = steam_dir()
    return steam / "config" / "steamvr.vrsettings" if steam is not None else None


def _driver_section(settings: dict) -> Optional[str]:
    wanted = f"driver_{DRIVER_NAME}".lower()
    return next((key for key in settings if key.lower() == wanted), None)


def driver_disabled_in_steamvr() -> bool:
    """SteamVR keeps add-ons the user (or safe mode, after a crash) turned off in steamvr.vrsettings."""
    path = _steamvr_settings_path()
    settings = _read_json(path) if path is not None else None
    if not settings:
        return False
    section = settings.get(_driver_section(settings) or "", {})
    return section.get("enable") is False or bool(section.get("blocked_by_safe_mode"))


def enable_driver_in_steamvr() -> bool:
    """Turn the add-on back on. Only while SteamVR is closed: it rewrites the file when it exits."""
    path = _steamvr_settings_path()
    settings = _read_json(path) if path is not None else None
    if not settings:
        return False
    key = _driver_section(settings)
    if key is None:
        return False
    section = settings[key]
    section["enable"] = True
    section.pop("blocked_by_safe_mode", None)
    temporary = path.with_suffix(".handcam-tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=3)
    os.replace(temporary, path)
    return True


def other_registrations(steamvr: Optional[Path]) -> List[Path]:
    """
    Copies of this driver registered elsewhere with vrpathreg: SteamVR loads one
    of them, maybe not the one the app installs.
    """
    found = []
    for folder in (Path(p) for p in _openvr_paths().get("external_drivers", [])):
        manifest = _read_json(folder / "driver.vrdrivermanifest") or {}
        if manifest.get("name") == DRIVER_NAME and (steamvr is None or folder != steamvr / "drivers" / DRIVER_FOLDER):
            found.append(folder)
    return found


# ----- the driver

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DriverStatus:
    state: str  # "no_steamvr", "missing", "outdated", "current"
    steamvr: Optional[Path] = None
    folder: Optional[Path] = None
    installed_version: Optional[str] = None
    bundled: bool = False
    disabled: bool = False
    elsewhere: List[Path] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        return self.bundled and (self.state in ("missing", "outdated") or self.disabled)


def driver_status(lay: Optional[Layout] = None, steamvr: Optional[Path] = None) -> DriverStatus:
    lay = lay or layout()
    steamvr = steamvr or find_steamvr()
    bundled = bool(lay.driver_files)
    if steamvr is None:
        return DriverStatus("no_steamvr", bundled=bundled)
    folder = steamvr / "drivers" / DRIVER_FOLDER
    status = DriverStatus("missing", steamvr, folder, bundled=bundled, elsewhere=other_registrations(steamvr))
    if not (folder / "bin" / "win64" / DRIVER_DLL).exists():
        return status
    status.disabled = driver_disabled_in_steamvr()
    status.installed_version = (_read_json(folder / DRIVER_VERSION_FILE) or {}).get("version")
    # Same files as this app carries, byte for byte, is up to date
    same = bundled and all((folder / rel).is_file() and _sha256(folder / rel) == _sha256(src)
                           for src, rel in lay.driver_files)
    status.state = "current" if same or not bundled else "outdated"
    return status


def install_driver(lay: Layout, steamvr: Path) -> Path:
    """Copy the bundled driver into SteamVR, replacing an older one, and make sure it is enabled."""
    if not lay.driver_files:
        raise AddonError("This copy of the app has no driver to install. From source, build it first "
                         "(INSTALL.md section 4).")
    if steamvr_running():
        raise AddonError("SteamVR is running. Close it completely (including the SteamVR status window) "
                         "and try again.")
    folder = steamvr / "drivers" / DRIVER_FOLDER
    try:
        for src, rel in lay.driver_files:
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        with open(folder / DRIVER_VERSION_FILE, "w", encoding="utf-8") as f:
            json.dump({"version": lay.version, "dll_sha256": _sha256(folder / "bin" / "win64" / DRIVER_DLL)}, f)
    except PermissionError as e:
        raise AddonError(f"Windows did not allow writing to {folder}. Start the app once with "
                         f"\"Run as administrator\" and try again.\n\n{e}") from e
    except OSError as e:
        raise AddonError(f"Could not copy the driver to {folder}:\n{e}") from e
    if driver_disabled_in_steamvr():
        enable_driver_in_steamvr()
    return folder


def uninstall_driver(steamvr: Path) -> None:
    if steamvr_running():
        raise AddonError("SteamVR is running. Close it completely and try again.")
    folder = steamvr / "drivers" / DRIVER_FOLDER
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        pass
    except OSError as e:
        raise AddonError(f"Could not remove {folder}:\n{e}") from e


# ----- WiLoR depth

@dataclass
class DepthStatus:
    state: str  # "missing", "outdated", "current"
    env: Path
    managed: bool  # installed by the app, so it can update and remove it
    installer: bool  # the installer script is available
    version: Optional[int] = None


def depth_status(lay: Optional[Layout] = None) -> DepthStatus:
    lay = lay or layout()
    here = Path(sys.executable).resolve().parent.parent == lay.depth_env.resolve()
    installer = lay.depth_installer.exists() and lay.depth_requirements.exists()
    status = DepthStatus("missing", lay.depth_env, lay.packaged, installer)
    if not (here or lay.depth_python.exists()):
        return status
    status.version = (_read_json(lay.depth_env / DEPTH_MARKER) or {}).get("version")
    if status.version is None and not lay.packaged:
        status.state = "current"  # set up by hand from source (INSTALL.md section 10)
    else:
        status.state = "current" if (status.version or 0) >= DEPTH_VERSION else "outdated"
    return status


def depth_install_command(lay: Layout) -> List[str]:
    """The installer, run without questions: the app asks for the license first."""
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(lay.depth_installer), "-Yes",
            "-Env", str(lay.depth_env), "-Requirements", str(lay.depth_requirements),
            "-Constraints", str(lay.depth_constraints), "-ComponentVersion", str(DEPTH_VERSION)]


def remove_depth(lay: Layout) -> None:
    if Path(sys.executable).resolve().parent.parent == lay.depth_env.resolve():
        raise AddonError("The app is running from the WiLoR environment. Restart it without WiLoR depth first.")
    try:
        shutil.rmtree(lay.depth_env)
    except FileNotFoundError:
        pass
    except OSError as e:
        raise AddonError(f"Could not remove {lay.depth_env}:\n{e}") from e


def gpu_summary() -> Optional[str]:
    """The NVIDIA GPU's name, memory and driver, or None without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    return line or None


def free_disk_gb(path: Path) -> Optional[float]:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return shutil.disk_usage(candidate).free / 1e9
    return None


def describe(status: DriverStatus, app_version: str) -> Dict[str, str]:
    """Plain words for a driver status: a headline and a detail line."""
    if status.state == "no_steamvr":
        return {"title": "SteamVR not found",
                "detail": "Install SteamVR from Steam, or pick its folder."}
    if status.state == "missing":
        return {"title": "Not installed", "detail": "Hands won't show in SteamVR until it is installed."}
    version = f"version {status.installed_version}" if status.installed_version else "an older version"
    if status.state == "outdated":
        return {"title": "Update available",
                "detail": f"SteamVR has {version} of the driver; this app brings version {app_version}."}
    title = "Installed, but turned off in SteamVR" if status.disabled else "Installed and up to date"
    return {"title": title, "detail": str(status.folder)}
