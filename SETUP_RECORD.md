# HandCameraDriver — Setup Record & Handoff

Working notes covering everything done to get this driver running, why, and what is
still outstanding. Written at the point where live hand tracking is confirmed working.

---

## 1. Current status

**Hand tracking works.** Confirmed by live diagnostic output in `vrserver.txt`:

```
HandTracking LEFT  pos: x=-0.064 y=-0.328 z=-0.654 trigger=1.00 grip=0.50 active=1
HandTracking RIGHT pos: x=-0.113 y=0.458 z=-0.714 trigger=0.00 grip=0.00 active=1
HandTracking RIGHT pos: x=-0.119 y=0.554 z=-0.700 trigger=0.00 grip=0.00 active=1
```

What this proves:

- Position values change in response to real hand movement
- `active=1` — the device activated correctly
- `trigger=1.00 grip=0.50` — the pinch gesture is detected and propagating
- No `Invalid handle` errors remain

**What is not done:** a stable 3D view in SteamVR. See section 8.

---

## 2. Environment

| Item | Value |
|---|---|
| OS | Windows, PowerShell 5.1 |
| Repo | `<repo>` |
| Python | 3.10.11 in repo-local `.venv` |
| SteamVR | `<Steam>\steamapps\common\SteamVR` |
| Driver install | `...\SteamVR\drivers\handcameradriver` |
| SteamVR config | `<Steam>\config\steamvr.vrsettings` |
| SteamVR log | `<Steam>\logs\vrserver.txt` |

Note: SteamVR is installed under Scoop, so paths differ from the default
`C:\Program Files (x86)\Steam\steamapps\common\SteamVR`. Adjust any command accordingly.

### Installed Python packages

```
mediapipe             0.10.21
numpy                 1.26.4
opencv-contrib-python 4.11.0.86
opencv-python         5.0.0.93   <- present but NOT the one imported
```

`opencv-python 5.0.0.93` is installed alongside `opencv-contrib-python 4.11.0.86`, but
the imported module resolves to **4.11.0**, which is what `requirements.txt` pins
(`opencv-python>=4.8.1.78,<5`). The 5.0.0.93 dist-info is leftover and can be removed
with `pip uninstall opencv-python` if you want a clean state. Verify with:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__)"
```

---

## 3. Pipeline

```
Phone camera
   │  (Iriun acts as a virtual webcam)
   ▼
Camera.py  ── MediaPipe extracts 21 landmarks per hand
   │
   ▼
TCP 127.0.0.1:65432   (loopback only, no auth)
   │
   ▼
hand_tracking_listener.cpp   (C++ side, inside vrserver)
   │
   ▼
SteamVR  →  two controller devices, position + trigger + grip
```

The driver exposes **two tracked controller devices** — one per hand — carrying position
and a trigger/grip axis driven by pinch detection.

**It is not skeletal hand tracking.** There is no per-finger joint data, no hand
orientation from finger spread, and no hand model. Anything expecting 21-joint skeletal
input (most current VR avatars and most hand-tracking apps) will not work with this
driver. It emulates controllers, not hands.

---

## 4. Changes made to the repo

All uncommitted. `git status` shows:

```
A  .gitmodules
 M README.md
 D SteamVR Driver/driver.vrdrivermanifest
 M SteamVR Driver/manifest/driver.vrdrivermanifest
 M SteamVR Driver/src/controller_device_driver.cpp
 M requirements.txt
A  third_party/openvr
?? SteamVR Driver/src/CMakeLists.txt
```

### 4.1 Build system (added)

`SteamVR Driver/src/CMakeLists.txt` did not exist. The repo shipped only Visual Studio
project files plus a legacy `DriverMain.cpp` that is an unsafe duplicate entry point.

- C++17
- Explicitly excludes `DriverMain.cpp`
- Finds OpenVR via the bundled SDK
- Target name `driver_HandTrackCamVR`
- Output to `SteamVR Driver/bin/win64`

The OpenVR SDK is now a pinned git submodule rather than an unversioned vendored copy:

```powershell
git submodule update --init --recursive
git submodule status   # expect tag v2.15.6
```

### 4.2 Manifest fixes

`SteamVR Driver/manifest/driver.vrdrivermanifest`:

- `resourcePath` corrected from `HandTrackCamVR/resources` to `resources`
- added `"alwaysActivate": true`

A stale duplicate at the repo root was deleted. SteamVR reads the manifest from the
**root of the driver folder**, so the installed copy has a root-level manifest even
though the repo keeps it under `manifest/`. If you reinstall from a clean copy, place the
manifest at the root of the installed driver directory.

### 4.3 Dependency bounds

`requirements.txt` now constrains majors:

```
opencv-python>=4.8.1.78,<5
mediapipe>=0.10.0,<1
numpy>=1.21.0,<2
```

MediaPipe does not support OpenCV 5 or NumPy 2.

### 4.4 Driver serial / settings keys

`controller_device_driver.cpp` read settings from the wrong section names and returned
empty model/serial strings. SteamVR rejected the device with
`Unable to init driver: No serial number`. Fixed to:

- sections `driver_hand_camera_tracking`, `..._left_hand`, `..._right_hand`
- keys `model_number` and `serial_number`
- zero-initialised char arrays with non-empty fallback strings
- corrected a left/right variable mix-up

### 4.5 Input handle guard (the important fix)

`MyRunFrame()` called `UpdateScalarComponent` and `UpdateBooleanComponent` on input
handles that `Activate()` had not yet created. SteamVR logged **12,142** errors like:

```
[Driver Input] UpdateScalarComponent failed. Invalid handle ulComponent: 0
[Driver Input] UpdateBooleanComponent failed. Invalid handle ulComponent: 0
```

Every trigger and grip update was being dropped. Guard added at the top of
`MyRunFrame()`:

```cpp
if ( !is_active_.load() || input_handles_[ MyComponent_trigger_value ] == vr::k_ulInvalidInputComponentHandle )
{
    return;
}
```

### 4.6 Diagnostics

Throttled to roughly one line per second per hand in `UpdateHandPosition()`:

```cpp
DriverLog( "HandTracking %s pos: x=%.3f y=%.3f z=%.3f trigger=%.2f grip=%.2f active=%d", ... );
```

This is your primary verification channel. Remove or gate behind a config flag before
any real use.

---

## 5. Build and deploy

Stop SteamVR first — `vrserver.exe` locks the DLL and the copy will fail with
*"The process cannot access the file ... because it is being used by another process."*

```powershell
Get-Process vrserver, vrstartup, vrcompositor -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

cd "<repo>\SteamVR Driver\src\build"
cmake --build . --config Release

Copy-Item `
  -LiteralPath "..\..\bin\win64\driver_HandTrackCamVR.dll" `
  -Destination "<Steam>\steamapps\common\SteamVR\drivers\handcameradriver\bin\win64\driver_HandTrackCamVR.dll" `
  -Force
```

DLL name must match the manifest `name` field: `driver_HandTrackCamVR.dll`. Mismatches
cause SteamVR to look for a file that does not exist.

SteamVR auto-discovers drivers from the `drivers` folder; no `vrpathreg` registration
is needed.

---

## 6. Running it

Order matters:

1. **Iriun** — start it first so the virtual webcam exists
2. **`Camera.py`** — reads camera index `0`, connects to the driver
3. **SteamVR** — the driver listens on `127.0.0.1:65432`

```powershell
.\.venv\Scripts\python.exe Camera.py
```

Enable the driver in **SteamVR → Settings → Startup/Shutdown → Manage Add-ons**.
It is not under Developer Settings.

Verify the socket:

```powershell
Get-NetTCPConnection -LocalPort 65432 -State Listen,Established
```

Both a `Listen` and an `Established` entry mean the Python side connected.

`config.json` currently: camera `device_id: 0`, 640×480 @ 60fps, `flip_horizontal: true`,
`max_hands: 2`, loopback `127.0.0.1:65432`.

---

## 7. Headsetless configuration

Applied to `config\steamvr.vrsettings` (backed up as `steamvr.vrsettings.bak`):

```json
"steamvr": {
    "requireHmd": false,
    "activateMultipleDrivers": true
},
"driver_null": {
    "enable": true
}
```

Also disabled monitor timeout so the desktop view stays awake:

```powershell
powercfg /change monitor-timeout-ac 0
```

### Why the null driver keeps going to sleep

`driver_null` is a simulated HMD with no sensors. It renders a mirror window, produces a
fixed pose, and becomes dormant without real hardware driving it. The mirror window
blanks after a short while. Windows itself is not sleeping the monitor — `VIDEOIDLE` is
now 0 and `STANDBYIDLE` was already 0.

Because controller poses in this driver are computed relative to the HMD pose
(`controller_device_driver.cpp`, `GetPose()` uses `GetRawTrackedDevicePoses`), hands
appear as two points offset from a phantom head at the world origin. That is expected
and still useful for verifying tracking, but they will not stay responsive indefinitely.

This is a limitation of simulated HMDs in general, not a fault in this driver.

---

## 8. Getting a real 3D session — phone as headset

This is the remaining piece.

### The conflict you need to know about

**The same phone cannot reliably be both.** Iriun is currently holding your phone's camera
to serve MediaPipe. If you simultaneously run a headset app (ALVR, Iriun VR, iVRy) on
that phone, two applications contend for the camera and the session degrades.

Realistic options:

- **Second phone** — cheapest and most reliable
- **PC webcam** for MediaPipe, phone dedicated as HMD — cleanest separation
- **Donated/cheap used VR headset** with ALVR — best quality, needs no phone

### Option A — ALVR with PhoneVR client (recommended open-source path)

ALVR officially lists PhoneVR support, though flagged as *"works on some smartphones, not
enough testing."*

1. Install SteamVR, launch once, close.
2. Download the ALVR Windows installer from <https://github.com/alvr-org/ALVR/releases>.
   Run as admin, keep the default install path.
3. On the phone: enable Developer Options, then sideload the **PhoneVR** client APK
   (<https://github.com/PhoneVR-Developers/PhoneVR/releases>).
   PhoneVR documents compatibility with ALVR Server v20.8.0; newer ALVR may work but is
   not guaranteed.
4. Launch ALVR on the PC and complete the setup wizard (firewall rules, encoder preset).
5. Phone and PC on the **same router**. 5 GHz for the phone, wired Ethernet for the PC.
   Expect roughly 150 ms latency.
6. Start ALVR client on the phone, accept **Trust** on the PC side, put the phone in a
   viewer, launch SteamVR from ALVR.

**Important conflict with this driver:** by default ALVR disables other SteamVR drivers
on startup to maximise compatibility. That would disable this hand-tracking driver. To
keep both, go to the ALVR **Installation** tab and click **Register ALVR driver** to
register it manually, which stops ALVR from disabling third-party drivers.

### ALVR hand tracking vs. this driver

ALVR has its own hand tracking, exposed to SteamVR as controller input. The gesture
mappings from the ALVR wiki:

| Gesture | Mapped input |
|---|---|
| Pinch thumb + index | Trigger |
| Curl thumb to palm | Joystick click |
| Curl middle, ring, little | Grip |
| Pinch thumb + middle | Y/B |
| Pinch thumb + ring | X/A |
| Pinch thumb + little (left hand) | Menu |
| Fist with thumb on top of hand | Joystick activation |

Recognition runs in the ALVR client, which covers both Quest and the PhoneVR client.

The distinction that matters here: ALVR produces **button and gesture values, not hand
position**. It tells SteamVR "trigger is pressed", never "your hand is at x=0.23,
y=-0.08, z=-0.66". This driver produces the opposite — MediaPipe derives hand position
and emits controller poses — with trigger and grip derived from pinch.

| | ALVR hand tracking | This driver |
|---|---|---|
| Output | Buttons / gestures | Controller position |
| Source | Headset or phone cameras | External camera + MediaPipe |
| Positional | No | Yes |
| Available with PhoneVR | Yes | Yes |

Neither replaces the other. ALVR covers face buttons and joystick, which this driver has
no equivalent for. This driver covers positional hands, which ALVR cannot produce.

If you enable both, two sources will contend for the same controller roles. Keep one
source per axis: let this driver own position, trigger, and grip, and do not rely on
ALVR's hand input for those same axes. Toggle ALVR gestures via
`Headset → Controllers → Gestures` if they conflict.

If SteamVR crashes on launch, ALVR resolution settings are the usual cause. Reset to
the headset's native resolution for both emulated and transcoded values. Foveated
rendering must be enabled.

### Option B — Iriun VR headset

Same vendor as the webcam you already use, so the tooling is familiar. Iriun VR is a
separate app from Iriun 4K Webcam.

- Install the Iriun driver on the PC
- Install the Iriun VR app on the phone
- Launch SteamVR, connect over local Wi-Fi

Note it is 3DoF-only on a flat phone screen with no tracking cameras. Adequate for
seated viewing; not usable for room-scale.

### Option C — iVRy

Commercial, has a native high-performance OpenVR driver, iOS and Android. No sideloading
required. Same 3DoF caveat.

### What each option gives you

| | HMD | Hands | Cost |
|---|---|---|---|
| Null driver (current) | none, sleeps | this driver | free |
| ALVR + PhoneVR | 3DoF, phone gyro | this driver | free |
| Iriun VR | 3DoF | this driver | free |
| Used Quest 2 + ALVR | 6DoF, real SLAM | this driver | ~$150–250 |

For anything beyond looking at the SteamVR dashboard, a 6DoF headset is the only
genuinely workable option. A phone is 3DoF at best and cannot give room-scale.

---

## 9. Socket input — fixed

The socket parser used unvalidated `std::stof` on received tokens, so malformed input
threw an uncaught `std::invalid_argument` and could terminate `vrserver.exe`. It now
parses with `std::from_chars` in `hand_message.cpp` and drops any malformed line (see
docs/PROTOCOL.md). The socket still has no authentication; it binds to loopback, so
only processes on the local machine can reach port 65432.

---

## 10. Repository hygiene

Not committed. When committing, note that `third_party/openvr` is a submodule and
`.gitmodules` must be included, and that `IMPLEMENTATION_SUMMARY.md` from earlier work
may now be partly stale.