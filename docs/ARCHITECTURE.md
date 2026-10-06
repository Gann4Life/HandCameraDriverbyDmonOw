# Architecture

How this project is built and the rules every change is reviewed against. When code and this
document disagree, either the code is fixed or this document is updated in the same change; never
leave them apart.

## Overview

Two programs that talk over a local socket:

- **Tracker (Python).** Reads a camera, finds the hands with MediaPipe, turns them into controller
  poses and inputs, and sends one line per hand per frame. Has a desktop GUI (`app.py`, PySide6)
  and a command-line entry (`Camera.py`).
- **Driver (C++, OpenVR).** A SteamVR driver that listens on the socket and presents the hands as
  two controllers (Touch or Index profile, with finger curls for the Index skeleton).

The line format between them is a contract, documented in [PROTOCOL.md](PROTOCOL.md). Neither side
changes it alone.

```
camera ─► CameraCapture ─► MediaPipe ─► HandIdentityTracker ─► hand_fit / pose ─► filters ─► hand_motion (MotionPredictor)
                                              │                                              ▲ hand_flow (lost hands)
                                              │
                                              └─► hand_features ─► ControlMapper / gesture_scores
                                                                          │
                          HandData.to_protocol_string ◄────────────────────┘
                                   │ socket
                                   ▼
              tracker_server ─► hand_tracking_listener ─► controller_device_driver ─► SteamVR
```

## Python layers

Each layer may only import from the layers below it.

| Layer | Modules | Knows about |
|---|---|---|
| Entry points | `app.py`, `Camera.py` `main()`, `calibrate.py` (legacy), `tools/` | Everything below |
| GUI | `gui/` (including `gui/environments.py`, which checks the add-ons) | Engine, config, presets, `addons`. Never MediaPipe or the socket directly |
| Engine | `Camera.HandTracker`, `gui/tracker_worker.py` (thread wrapper), `depth_assist`, `session_recorder` | Config, domain, I/O |
| Config | `config_defaults`, `presets`, `builtin_presets`, `utils/config_utils` | Plain dicts. `config_defaults` collects each domain module's own `DEFAULTS` |
| Domain | `hand_features`, `hand_controls`, `gesture_scores`, `hand_fit`, `hand_identity`, `hand_motion`, `hand_flow` (uses OpenCV for the flow, no windows), `hand_data`, `gesture_detector` (legacy) | Pure math and numpy. No Qt, no OpenCV windows, no sockets, no files |
| I/O and platform | `utils/camera_utils`, `utils/socket_client`, `utils/win_process`, `addons` | The outside world |
| Utilities | `utils/one_euro`, `utils/hand_size`, `version` | Nothing project-specific |

Rules that follow from this:

- **Domain code is testable without a camera, a GPU, a window or SteamVR.** If a test needs one of
  those, the logic is in the wrong layer.
- **Both hands share one implementation.** State is keyed by hand; there is never a left and a right
  copy of the same logic. Differences between hands come from the data, not from code paths.
- **Settings are data.** A new setting is one entry in `gui/settings_schema.py` plus its default in
  the module that owns it; the GUI and live apply follow from that. (Not yet true for settings that
  rebuild something: see Known gaps.)
- **Machine-local settings stay out of presets** (camera device, URLs, paths): presets ship as
  built-in defaults on every release.

## Design principles

Applied with judgement, not mechanically. When two pull in opposite directions (most often SOLID vs
KISS), the simpler option wins unless the other one removes a real, present problem; the review says
which trade-off was made and why.

- **Single responsibility.** A module or class has one reason to change. A name that needs "and"
  (`detect gestures and compute orientation`) is a split waiting to happen.
- **Open/closed.** Extend through data or new types (settings schema, gesture table, backends)
  instead of growing `if mode == ...` chains.
- **Dependency inversion.** The engine receives its collaborators (camera, sender, filters) rather
  than reaching for globals, so tests can swap them.
- **DRY.** One source of truth per fact: a default lives in one place, a protocol field is written in
  one place and parsed in one place. Duplication across the Python/C++ boundary is checked against
  PROTOCOL.md.
- **KISS / YAGNI.** No abstraction for a second case that doesn't exist yet. Three similar lines beat
  a premature framework.
- **Modularity.** Small modules with a clear public surface. Private helpers start with `_`.
- **Composition over inheritance.** Inheritance only for real "is-a" relations (Qt widgets, OpenVR
  interfaces).

## Code style

- Readable names over comments. Comments explain **why**, never restate the line below.
- Docstrings on public modules, classes and non-obvious functions. They must stay true; a wrong
  docstring is a bug.
- Type hints on public functions. Units in names or docstrings (`_m`, `_deg`, `_s`).
- Python follows PEP 8 with a 120-column limit. C++ follows the style already in `SteamVR Driver/src`.
- No dead code, no commented-out code, no `print` debugging left behind (use the log).

## Driver (C++)

- `DriverMain.cpp` / `device_provider` register the devices; `controller_device_driver` owns one
  controller and is instantiated once per hand (same class, never a left and a right copy).
- `tracker_server` owns the socket (no OpenVR, tested with real sockets). It polls with `poll`
  (`WSAPoll` on Windows) instead of blocking, so `Stop` never closes a socket under the thread. Only its thread touches the
  client socket. An accept error doesn't end listening, and an exception while serving a connection
  closes that connection and the server keeps listening. On Windows it binds with
  `SO_EXCLUSIVEADDRUSE`, so no other process can share the port. It serves a connection only after
  the tracker's greeting and closes one that doesn't greet within 1 s, doesn't open with the
  greeting, has another protocol version, or sends no complete line for 5 s after the greeting
  (the tracker's keepalive repeats the greeting, which the server ignores). Per-connection log lines
  are limited to one a second and each refusal reason is logged once (see PROTOCOL.md).
- `hand_tracking_listener` only applies messages to the controllers. Line framing and parsing live
  in `hand_message` (no OpenVR, unit-tested): it hands over complete, validated messages and never
  partial ones.
- A controller's pose is stored as one struct under a mutex, so the position and rotation are never
  read torn.
- Input from the socket is untrusted: every field is validated and clamped; a malformed line is
  dropped, never crashes the driver or SteamVR.
- No per-frame logging in release builds.

## Performance

- Tracking runs on its own thread; the GUI never blocks it, and it never blocks the GUI.
- The per-frame budget is set by the camera (about 33 ms at 30 fps). Work that doesn't fit runs
  asynchronously on the newest frame, never by queueing old ones.
- Session recordings are written on background threads behind bounded queues. If the disk falls
  behind, frames are left out of the video (their data is kept). Only if the data queue also fills are records dropped, and counted in the
  `end` record. Tracking is never delayed.
- No optimisation without numbers: measure before and after (per-frame latency, fps, CPU/GPU), on the
  same recorded clip when possible, and include the numbers in the change.

## Testing

Python tests live in `tests/` (pytest). Install the dev requirements once with
`python -m pip install -r requirements-dev.txt`, then run `python -m pytest -q` from the repo root.
The driver has its own tests in `SteamVR Driver/src/tests/`, built with the driver: the line parser
and greeting (`hand_message_tests`) and the socket server (`tracker_server_tests`, real sockets).
Run both with `ctest --test-dir "SteamVR Driver/src/build" -C Release`.

- Every bug fix starts with a test that reproduces it. Every feature ships with its tests in the same
  change.
- Domain code gets unit tests; GUI code gets offscreen smoke tests (`QT_QPA_PLATFORM=offscreen`).
- Tracking regressions are checked on recorded clips, so results repeat.
- Anything that can only be checked in VR is listed as a manual test; a VR fix is not done until it
  passes in VR.

## Workflow

- Branches: `main` (releases), `dev` (pre-release). Every change gets its own branch from `dev`
  (`feat/...`, `fix/...`, `docs/...`, `chore/...`, `test/...`, `refactor/...`) and merges back into `dev` with `--no-ff`. Only
  `dev` merges into `main`.
- Conventional Commits.
- Before merging into `dev`: tests pass, the code review and the docs update are done.

## Known gaps

Where the code doesn't follow this document yet. Each one becomes an issue; remove the line when it's
fixed.

- `Camera.py` holds both the engine (`HandTracker`, ~900 lines: config loading, camera, pose, depth,
  filters, drawing, CLI preview) and the command-line entry. It needs splitting by responsibility,
  and `HandTracker` builds its own camera, MediaPipe model and socket instead of receiving them, so
  none of its pose logic can be tested. That includes `HandTracker.predict_lost_hands` and the
  blend back to a found hand, and how `hand_flow` is fed frames: they have no unit tests, only
  `hand_motion` and `hand_flow` themselves do.
- Defaults are repeated: `Camera.py` has about 25 `.get(key, default)` fallbacks that duplicate
  `DEFAULT_CONFIG` (and `TrackingFrame.hfov_deg` another).
- Which settings rebuild the camera, model or filters lives twice: `Camera.setting_action` (an
  `if key.startswith` chain) and the `Apply` labels in `gui/settings_schema.py`.
- The Index controller placement is corrected on both sides: `HandTracker.index_adjustment` (tuned
  live, `network.index_*`) on top of a fixed Touch→Index transform in `controller_device_driver.cpp`.
  One side should own it.
- The socket's 5 s timeout also applies to `sendall` on the tracking thread, so a hung SteamVR can
  stall tracking.
- `gesture_detector.py` mixes quaternion math, legacy on/off gestures and handedness evidence. The
  quaternion helpers belong in a math module; the legacy gestures are superseded by `gesture_scores`.
  `hand_motion` imports those helpers and adds rotation-vector helpers of its own, which belong in
  the same module.
- `calibrate.py` is a standalone OpenCV script that predates the GUI.
- The driver parses `GESTURE:` but ignores it, so gestures never reach the game.
- The WiLoR depth experiment (`depth_assist.py`, `gui/environments.py`, its installer and add-on) is
  planned for removal.
- Tests cover only part of the domain code so far (`hand_features`, `hand_controls`, `gesture_scores`,
  `utils/one_euro`, `HandData.to_protocol_string`, `utils/socket_client`, `hand_identity`, `hand_motion`,
  `hand_flow`, `presets` migration, one recorded frame of `hand_fit`) and the driver's line
  parser and socket server. There are no GUI smoke tests, recorded-clip tests or tests of the
  driver's pose logic yet.
