# Installation Guide (Windows)

Step-by-step setup, from a clean Windows PC to seeing your hands in SteamVR.
Tested on Windows 10 with Python 3.10 and SteamVR 2.x.

How the pieces fit together:

```
camera ──► Camera.py (Python, MediaPipe) ──TCP 127.0.0.1:65432──► SteamVR driver (C++ DLL) ──► games
```

`Camera.py` finds your hands in the camera image. The driver shows them to SteamVR as
two **Oculus Touch controllers**, so any game that supports Touch controllers will show them.

---

## 1. What you need

### Hardware
- **A SteamVR headset**: a PC VR headset, or a Quest streaming through ALVR, Virtual
  Desktop or Link.
- **A camera** that delivers at least 30 fps at 640×480. Either of these works:
  - A phone running [Iriun Webcam](https://iriun.com/). Connect it over **USB** for lower
    latency than Wi-Fi.
  - Any USB webcam.
- **A spot for the camera.** There are two modes:
  - **POV** (the default): the camera is on your head or chest and looks where you
    look, so it sees the backs of your hands.
  - **Facing**: the camera is in front of you, looking at you.

### Software to install
| Software | Why | Where |
|---|---|---|
| Steam + SteamVR | The VR runtime that loads the driver | Steam store, search "SteamVR" |
| Git | To download the repo and its OpenVR submodule | https://git-scm.com/download/win |
| Python **3.10** (64-bit) | Runs the tracker. 3.11 and 3.12 should also work, but only 3.10 is tested. | https://www.python.org/downloads/ (tick "Add python.exe to PATH") |
| Visual Studio 2022 (or newer) **Build Tools** | Compiles the driver | https://visualstudio.microsoft.com/downloads/ → "Build Tools for Visual Studio" |
| Camera app (optional) | Only if you use a phone as the camera | Iriun Webcam on the PC **and** on the phone |

When you install the Visual Studio Build Tools, choose the **"Desktop development with C++"**
workload. It already includes the MSVC compiler, the Windows SDK and CMake, so you don't
need to install CMake separately.

---

## 2. Download the code

Open **PowerShell** and run:

```powershell
git clone --recursive https://github.com/Gann4Life/HandCameraDriverbyDmonOw.git
cd HandCameraDriverbyDmonOw
```

`--recursive` is required: it downloads the OpenVR SDK (`third_party/openvr`), and the
driver can't be built without it. If you already cloned without it, run:

```powershell
git submodule update --init --recursive
```

All commands below run from this folder.

---

## 3. Set up Python

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Check that it worked:

```powershell
.venv\Scripts\python -c "import cv2, mediapipe, numpy; print(cv2.__version__, mediapipe.__version__, numpy.__version__)"
```

You should see OpenCV `4.x`, MediaPipe `0.10.21` and NumPy `1.x`. MediaPipe needs NumPy 1
and OpenCV 4, and newer MediaPipe releases dropped the API this project uses, which is
why `requirements.txt` pins those versions.

---

## 4. Compile the driver

Open **"Developer PowerShell for VS"** from the Start menu (a normal PowerShell works too,
as long as `cmake` is on PATH). From the repo folder:

```powershell
cmake -S "SteamVR Driver\src" -B "SteamVR Driver\src\build" -A x64
cmake --build "SteamVR Driver\src\build" --config Release
```

The result is **`SteamVR Driver\bin\win64\driver_HandTrackCamVR.dll`**.

If CMake says *"OpenVR SDK headers were not found"*, the submodule is missing. Run the
`git submodule update` command from step 2.

---

## 5. Install the driver into SteamVR

### 5.1 Find your SteamVR folder
By default it's `C:\Program Files (x86)\Steam\steamapps\common\SteamVR`. If you put Steam
somewhere else: in Steam, go to **Library → SteamVR → right-click → Manage → Browse local files**.

### 5.2 Copy the files
**Close SteamVR first.** While it runs it locks the DLL and the copy fails. Then, in
PowerShell from the repo folder (change `$steamvr` if your path is different):

```powershell
$steamvr = "C:\Program Files (x86)\Steam\steamapps\common\SteamVR"
$dest = "$steamvr\drivers\handcameradriver"

New-Item -ItemType Directory -Force "$dest\bin\win64" | Out-Null
Copy-Item "SteamVR Driver\manifest\driver.vrdrivermanifest" "$dest\driver.vrdrivermanifest" -Force
Copy-Item "SteamVR Driver\bin\win64\driver_HandTrackCamVR.dll" "$dest\bin\win64\" -Force
Copy-Item "SteamVR Driver\resources" "$dest\" -Recurse -Force
```

The installed folder must look like this. The manifest goes at the **root**, not in a
`manifest\` subfolder:

```
drivers\handcameradriver\
├── driver.vrdrivermanifest
├── bin\win64\driver_HandTrackCamVR.dll
└── resources\...
```

### 5.3 Enable it
1. Start SteamVR.
2. Go to **Settings → Startup / Shutdown → Manage Add-ons** and turn **HandTrackCamVR** on.
3. Restart SteamVR.

SteamVR turns add-ons off automatically after a crash, so if the hands stop appearing
some day, check here first.

### 5.4 If you use ALVR
ALVR turns off other SteamVR drivers by default. In ALVR, open the **Installation** tab and
click **Register ALVR driver**: that stops ALVR from turning this one off. Also turn off
ALVR's own hand tracking and controller emulation
(**Headset → Controllers**). Otherwise two sources fight over the same hands.

---

## 6. Set up the camera

1. If you use a phone, start **Iriun** on the PC and on the phone, and connect the phone over USB.
2. Open `config.json` and set the `camera` section:
   - `device_id`: the camera's index. `0` is usually the built-in webcam, and
     virtual cameras like Iriun are usually `1` or `2`. Try each one until the preview
     shows the right camera.
   - `width`, `height`, `fps`: `640`, `480`, `30` is a safe start.
   - `hfov_deg`: the camera's horizontal field of view in degrees. It's used to turn the
     image into real distances. Phone main cameras are about 65–75°; the default is 70.
   - `source_mirrored`: set it to `true` only if your camera app already mirrors the
     image like a selfie. Iriun doesn't by default, so leave it at `false`.
   - `rotate_180`: set it to `true` if the camera is mounted upside down.
3. Pick the preset for your camera position (see section 1): in the app, **Preset** at the top
   of the Settings tab; in `config.json`, `"preset": "POV"` or `"preset": "Facing"`.

---

## 7. Run it

The order doesn't matter. `Camera.py` keeps retrying until the driver shows up.

1. Start SteamVR (with the headset connected).
2. Start the tracker app:
   ```powershell
   .venv\Scripts\python app.py
   ```
   It shows the camera with the hand skeletons, a 3D view of where each hand is placed, a live
   readout of what each hand sends, and all the settings. Most settings apply instantly while you
   watch the preview; the few that need more say so next to their name (for example "reopens the
   camera"). **Save** (Ctrl+S) writes them to `config.json`; **Revert** goes back to the last save.
   Hover over a setting's name for a short explanation.

   The command-line version still works, with an OpenCV preview window and keys instead of a
   settings panel:
   ```powershell
   .venv\Scripts\python Camera.py
   ```
   Options:
   - `--preset NAME`: use that preset (`POV`, `Facing` or one of yours) instead of the last one used.
   - `--mode pov` or `--mode facing`: use that mode's built-in preset.
   - `--swap-hands`: if left and right come out reversed.
   - `--rotate-180`: if the camera is mounted upside down.
3. A preview window opens with the hand skeletons drawn on it. In SteamVR you should see
   two Oculus Touch controllers that follow your hands.

The tracker keeps running at full speed in the background, even when the game has focus.

### Keys (click the preview window first)
| Key | What it does |
|---|---|
| `q` | Quit |
| `s` | Swap left/right **now**, if the tracker gets the hands the wrong way round |
| `f` | Cycle smoothing: One Euro → EMA → none |
| `b` | Toggle WiLoR depth (experimental, see section 10) |

### Is it working?
- **Preview:** you should see the skeleton drawn on each hand, plus `LEFT: ... NN cm` /
  `RIGHT: ... NN cm` lines at the top left showing each hand's distance from the camera.
- **Connection:** `Get-NetTCPConnection -LocalPort 65432` should show an `Established` entry.
- **Driver log:** the SteamVR log (`Steam\logs\vrserver.txt`) should have about one
  `HandTracking LEFT/RIGHT pos: ...` line per second.

---

## 8. Calibrate

In `app.py`, all of these are in the Settings tab and apply live, so you can adjust them while
looking at the 3D view and the cm readout. With `Camera.py`, edit `config.json` and restart it.

### Presets
Each camera position keeps its own settings in a **preset**: view, mirroring, placement,
smoothing, gestures and hand identity. Switching preset puts all of them back, so tuning one
position never undoes another. Settings marked *(all presets)* belong to the camera hardware or
the whole app (camera index, resolution, field of view, tracking model, depth source, controller
rotation, driver connection) and are the same in every preset.

- **POV** and **Facing** are built in. You can change them, and **Restore default** puts them
  back as shipped; they can't be renamed or deleted.
- **Duplicate...** makes a new preset from the settings in use, e.g. for a second camera
  position. Your own presets can be renamed and deleted.
- **Save** stores the settings in use in the active preset. A *modified* tag next to the preset
  means there are changes it doesn't have yet.

Presets live in `config.json`: `preset` is the active one and `presets` holds yours, plus any
built-in one you changed. Older configs with `calibration.pov` / `calibration.facing` are
converted on load.

### What to change
| Problem in VR | What to change |
|---|---|
| Hands are rotated compared to your real hands | `rotation_offset_deg.left` / `.right`, in degrees `[x, y, z]`. A POV camera usually needs a z value: the reference setup uses `-127` for the left hand and `127` for the right. Adjust in steps of 10–20°. |
| Hands are shifted (too high, too far…) | `position_offset`, in metres `[x, y, z]`. |
| Camera is tilted compared to your head | `camera_rotation_deg`. |
| Hands are too close or too far | Hold your wrist at a measured distance (e.g. 40 cm) and compare it with the cm the overlay shows. Fix it with `camera.hfov_deg` first, then `calibration.hand_scale`. |
| Hands feel shaky or laggy | Press `f` to compare the filters, then adjust `calibration.filter`. A lower `min_cutoff` gives more smoothing; a higher `beta` gives less lag on fast moves. |

Don't use `calibrate.py`: it's older than the app and only sets `position_offset` and `scale`;
the Settings tab covers both, with a live preview.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Left and right are swapped all the time | Start with `--swap-hands`, or set `tracking.swap_hands: true`. If it only happens now and then, press `s`. |
| Preview window is black, or shows the wrong camera | Wrong `device_id`. Also close any other app that's using the camera. |
| "Camera stopped delivering frames" | The tracker reconnects on its own for up to 30 s (`camera.reconnect_timeout`). With a phone, check the USB cable or Wi-Fi and keep the Iriun app in the foreground. |
| Hands don't show in SteamVR | Is the add-on enabled (5.3)? Is the manifest at the folder root (5.2)? Search `vrserver.txt` for `HandTrackCamVR`. |
| Controllers show in SteamVR but not in the game | Enable controllers/hands in the game's own settings. Turn off your headset's real controllers, since they take the hand slots too. |
| Copying the DLL fails ("used by another process") | Close SteamVR completely, including `vrserver.exe`, and try again. |
| Low FPS | Use `model_complexity: 0` and connect the phone over USB. The overlay shows the camera FPS and the processing FPS separately, so you can tell which one is the bottleneck. |

---

## 10. Optional: WiLoR depth (experimental)

This mode uses a 3D hand-mesh model to get much steadier distance from the camera, at the
cost of about 200 ms of extra depth lag.

What it needs:
- An NVIDIA GPU with about 1.5 GB of free VRAM.
- About 5 GB of disk space.
- Python 3.10.

> **⚠️ License: personal, non-commercial use only.** WiLoR is CC BY-NC-ND, the MANO hand
> model is non-commercial and non-redistributable, and the YOLO detector (Ultralytics) is
> AGPL-3.0. Don't redistribute it or use it commercially without legal review.

Install it into a **separate** environment, so the normal `.venv` stays as it is:

```powershell
py -3.10 -m venv .venv-wilor
.venv-wilor\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv-wilor\Scripts\python -m pip install "numpy<2" setuptools wheel
.venv-wilor\Scripts\python -m pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
.venv-wilor\Scripts\python -m pip install --no-build-isolation -r requirements.txt -r requirements-wilor.txt "numpy<2"
```

Then run `.venv-wilor\Scripts\python app.py` and turn on **WiLoR depth** in the toolbar (it
asks for confirmation and lists the costs first), or run `.venv-wilor\Scripts\python Camera.py`
and press `b`. While it is on, the status bar shows its rate and GPU memory. The models download the first
time, which takes a while. After that, loading takes about 10 s. When it's on, the overlay
shows `Depth [b]: WiLoR N Hz`.

To calibrate it: hold your wrist 40 cm from the camera and adjust
`tracking.depth_assist.scale` until the overlay reads `40 cm`. If the game starts dropping
frames, lower `tracking.depth_assist.max_rate_hz`.
