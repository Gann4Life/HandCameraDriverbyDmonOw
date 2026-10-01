"""
Windows process tuning so tracking keeps full speed while another window
(the VR game) has focus.
"""
import ctypes
import sys
from ctypes import wintypes


PRIORITY_CLASSES = {
    "normal": 0x00000020,        # NORMAL_PRIORITY_CLASS
    "above_normal": 0x00008000,  # ABOVE_NORMAL_PRIORITY_CLASS
    "high": 0x00000080,          # HIGH_PRIORITY_CLASS
}

# SetProcessInformation / PROCESS_POWER_THROTTLING_STATE
_PROCESS_POWER_THROTTLING = 4
_POWER_THROTTLING_CURRENT_VERSION = 1
_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG)]


def _set_power_throttling_off(kernel32, process, control_mask: int) -> bool:
    # ControlMask selects which policies this call governs; a StateMask of 0
    # turns all of them off, i.e. opts the process out of throttling
    state = _PowerThrottlingState(_POWER_THROTTLING_CURRENT_VERSION, control_mask, 0)
    return bool(kernel32.SetProcessInformation(process, _PROCESS_POWER_THROTTLING,
                                               ctypes.byref(state), ctypes.sizeof(state)))


def keep_running_in_background(priority: str = "above_normal",
                               disable_power_throttling: bool = True) -> list:
    """
    Raise this process's priority class and opt it out of background power
    throttling, so a focused, CPU-heavy game does not starve it.

    Args:
        priority: "normal", "above_normal" or "high"
        disable_power_throttling: Opt out of EcoQoS / power throttling

    Returns:
        Human-readable lines describing what was applied, for the console
    """
    if sys.platform != "win32":
        return []

    report = []
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    process = kernel32.GetCurrentProcess()

    priority_class = PRIORITY_CLASSES.get(str(priority).lower())
    if priority_class is None:
        report.append(f"Warning: unknown process priority '{priority}', leaving it unchanged")
    elif kernel32.SetPriorityClass(process, priority_class):
        report.append(f"Process priority: {priority}")
    else:
        report.append(f"Warning: could not set process priority (error {ctypes.get_last_error()})")

    if disable_power_throttling:
        # The timer-resolution flag only exists on Windows 11; older systems
        # reject the whole call if it is included, so fall back without it
        if _set_power_throttling_off(kernel32, process, _POWER_THROTTLING_EXECUTION_SPEED |
                                     _POWER_THROTTLING_IGNORE_TIMER_RESOLUTION):
            report.append("Power throttling: off (execution speed, timer resolution)")
        elif _set_power_throttling_off(kernel32, process, _POWER_THROTTLING_EXECUTION_SPEED):
            report.append("Power throttling: off (execution speed)")
        else:
            report.append(f"Warning: could not disable power throttling (error {ctypes.get_last_error()})")

    return report
