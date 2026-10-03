# Installation Guide (Windows)

Step-by-step setup, from a clean Windows PC to seeing your hands in SteamVR.
Tested on Windows 10 with Python 3.10 and SteamVR 2.x.

How the pieces fit together:

```
camera ──► Camera.py (Python, MediaPipe) ──TCP 127.0.0.1:65432──► SteamVR driver (C++ DLL) ──► games
```

`Camera.py` finds your hands in the camera image. The driver shows them to SteamVR as
two **Oculus Touch controllers**, so any game that supports Touch controllers will show them.
It can show them as **Valve Index controllers** instead (Settings → Driver connection →
*Show hands as*, or `network.controller_type: "index"`): then games that support Index finger
tracking also show each finger. Switching takes effect the next time SteamVR starts.

> **Don't want to compile anything?** Download the ready-to-run zip from
> [Releases](https://github.com/Gann4Life/HandCameraDriverbyDmonOw/releases), extract it and
> run `HandCameraDriver.exe`. The app installs and updates its SteamVR driver itself
> (**Add-ons** in the toolbar), so you can skip sections 2–5. This guide is for running from source.

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
  - **Facing** (the default, recommended): the camera is in front of you, looking at you.
  - **POV** (experimental): the camera is on your head or chest and looks where you
    look, so it sees the backs of your hands. Seen from behind, the tracker's own 3D hand folds
    and twists, so this preset rebuilds each hand from where its joints appear in the picture
    (**Rebuild hands in 3D**, in the Depth section).

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

The easy way: close SteamVR, start the app (`.venv\Scripts\python app.py`, section 7) and click
**Add-ons** in the toolbar. It finds SteamVR, shows whether the driver is installed and up to date,
and installs or updates it with one button; it also turns the driver back on if SteamVR disabled
it. The app checks this when it starts and offers it when something is missing or older than
the driver you built. The steps below do the same by hand.

### Updating the driver
After pulling new code, rebuild the driver (section 4), close SteamVR and click **Update** in
**Add-ons**, or copy the files again as in 5.2. With the ready-to-run zip, the new app offers the
update on its own the first time it starts.

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
   - `source_mirrored`: whether the picture already arrives mirrored like a selfie. The Facing
     preset turns it on, which is what the reference setup needs; if your hands move the
     wrong way left and right, switch it.
   - `rotate_180`: set it to `true` if the camera is mounted upside down.
3. Pick the preset for your camera position (see section 1): in the app, **Preset** at the top
   of the Settings tab; in `config.json`, `"preset": "Facing"` (the default) or `"preset": "POV"`.

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
   camera"). Hover over a setting's name for a short explanation. The settings are on two tabs:
   - **Preset**: what depends on where the camera is (section 8). Changes stay unsaved until
     **Save preset** (Ctrl+S); **Discard changes** goes back to the preset as saved.
   - **App settings**: the same for every preset (camera index, resolution, tracking model,
     depth, driver connection). These are saved to `config.json` as soon as you change them.

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
   two Oculus Touch (or Index) controllers that follow your hands. They appear once the tracker
   connects, so SteamVR shows no controllers from this driver until it is running.

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

In `app.py`, all of these are in the Preset and App settings tabs and apply live, so you can
adjust them while looking at the 3D view and the cm readout. With `Camera.py`, edit `config.json`
and restart it.

### Presets
Each camera position keeps its own settings in a **preset**: view, mirroring, placement
(position offset, camera tilt, hand rotation), smoothing, gestures and hand identity. Switching preset puts all of them back, so tuning one
position never undoes another. They are the settings on the **Preset** tab. The **App settings**
tab has the ones that belong to the camera hardware or the whole app (camera index, resolution,
field of view, tracking model, depth source, driver connection): they are the same in every
preset and save themselves.

- **Facing** (the default), **POV** (experimental) and **POV Pointer** (POV with the hands turned
  for pointing) are built in. You can change them, and **Restore default** puts them back as
  shipped; they can't be renamed or deleted.
- **Duplicate...** makes a new preset from the settings in use, e.g. for a second camera
  position. Your own presets can be renamed and deleted.
- **Save preset** stores the settings in use in the active preset. An *unsaved changes* tag next to
  the preset means there are changes it doesn't have yet; **Discard changes** drops them.
  Switching preset or closing the app with unsaved changes asks first.

Presets live in `config.json`: `preset` is the active one and `presets` holds yours, plus any
built-in one you changed. Older configs with `calibration.pov` / `calibration.facing` are
converted on load.

### What to change
| Problem in VR | What to change |
|---|---|
| Hands are rotated compared to your real hands | `rotation_offset_deg.left` / `.right`, in degrees `[x, y, z]`. The built-in presets come tuned on the reference setup: Facing uses `[0, 0, -90]` for the left hand and `[0, 0, 90]` for the right; POV uses `[0, 35, -115]` and `[0, -35, 115]`. Adjust in steps of 10–20°. |
| Hands are shifted (too high, too far…) | `position_offset`, in metres `[x, y, z]`. |
| Camera is tilted compared to your head | `camera_rotation_deg`. |
| Hands are too close or too far | Hold your wrist at a measured distance (e.g. 40 cm) and compare it with the cm the overlay shows. Fix it with `camera.hfov_deg` first, then `calibration.hand_scale`. |
| Hands feel shaky or laggy | Press `f` to compare the filters, then adjust `calibration.filter`. A lower `min_cutoff` gives more smoothing; a higher `beta` gives less lag on fast moves. |
| Hands drift toward and away from you | Turn on **Steady hand size** (`calibration.steady_hand_size`, on in the POV preset). It holds each hand at its recent median size, so its distance stops wobbling. |
| Hands twist, fold or flip when the camera sees their backs | Turn on **Rebuild hands in 3D** (`calibration.rebuild_hand`, on in the POV preset). Each hand is rebuilt from its joints in the picture with a hand model whose fingers only bend toward the palm, which steadies distance, rotation and gestures. It costs a few milliseconds per hand. |

Don't use `calibrate.py`: it's older than the app and only sets `position_offset` and `scale`;
the Settings tab covers both, with a live preview.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Left and right are swapped all the time | Start with `--swap-hands`, or set `tracking.swap_hands: true`. If it only happens now and then, press `s`. |
| Preview window is black, or shows the wrong camera | Wrong `device_id`. Also close any other app that's using the camera. |
| "Camera stopped delivering frames" | The tracker reconnects on its own for up to 30 s (`camera.reconnect_timeout`). With a phone, check the USB cable or Wi-Fi and keep the Iriun app in the foreground. |
| Hands don't show in SteamVR | Is the tracker running? The controllers only appear once it connects. Open **Add-ons** in the app: it shows whether the driver is installed, up to date and turned on. By hand: is the add-on enabled (5.3)? Is the manifest at the folder root (5.2)? Search `vrserver.txt` for `HandTrackCamVR`. |
| Changed *Show hands as* but SteamVR still shows the old controllers | Restart SteamVR. The driver picks the type from the first message after SteamVR starts. |
| With Index, the game ignores the hands or the fingers don't move | Not every game reads Index finger tracking. Switch *Show hands as* back to Touch and restart SteamVR. |
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

The easy way is **Add-ons** in the app: it shows the license notice, then downloads and installs
everything into its own environment (`.venv-wilor` from source, `depth\.venv` next to the
ready-to-run exe), with progress in the window. The same installer runs by hand:

```powershell
powershell -ExecutionPolicy Bypass -File installers\install-depth.ps1 -Env .venv-wilor `
  -Requirements requirements.txt -Constraints installers\depth-constraints.txt
```

Or set it up step by step, in a **separate** environment so the normal `.venv` stays as it is:

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
