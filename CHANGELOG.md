# Changelog

What changed for users in each version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Hands follow** setting (View): with a camera in front of you, choose *The room* and your hands
  keep their place while you look around, instead of turning with your head. They still move with
  you when you walk, and turn with your game's snap turn.
- **Recenter hands** (toolbar or R): after a 3 s countdown to face the camera, the room-anchored
  hands are turned to match where the camera is. Only the heading counts, so a tilted head doesn't
  tilt the hands.
- **Calibrate gestures** (toolbar): records your open hand and your fist and fits each finger's
  curl range to the active preset's view.
- **Session recording** (toolbar **Record** or F9): saves the camera video and, next to it, a data
  file with what the tracker made of every frame, to the `.output` folder next to the app. Use it to
  study a problem or replay it later. **File > Record the camera video** (on by default) turns the
  video off to keep only the data. **File > Open recordings folder** opens the folder. The video
  shows you and your room, so check it before you share it. The data file has paths, URLs and your
  user name removed from the settings and the log. Recording never slows tracking: if the disk can't
  keep up, frames are left out of the video, not delayed.

### Changed
- The grip now holds by itself. Once it reaches **Grip holds from**, it keeps its peak until it
  stays under **Grip lets go below** for 0.1 s, so a value near the game's threshold no longer
  grabs and drops. Below **Grip holds from** the grip is analog, and the trigger is always analog. This replaces the
  **Steady grip** and **Steady trigger** settings (they snapped to 0 or 1 and dropped quick
  pinches). Saved presets lose the old switches on their own.
- The app and the SteamVR driver now greet each other with a protocol version when they connect. The
  driver refuses a connection that doesn't greet within 1 s, isn't from the app (another program, a
  browser), or has a different version, and logs why. If you update only one side, the log says which
  one to update; the Add-ons window updates the driver. An older app can't connect to this driver.
- The driver closes a connection that sends nothing for 5 s. The app keeps it open while no hands are
  visible.
- The SteamVR driver is the only program that can use its port on Windows.

### Fixed
- Left and right are steadier, with one hand or two:
  - A single hand in view keeps its side however far it moves across the picture (a turning
    camera) and when it is briefly out of view.
  - In POV, seeing the back of the hand no longer flips it to the other side.
  - A second hand, or part of one at the edge of the picture, no longer pushes the tracked hand to
    the other side.
  - A real change of side now needs about half a second of clear evidence instead of a fifth.
    Every preset with the old value moves to the new one, yours included; a value you changed
    yourself stays.
  - A hand raised right after the other one was lowered is placed by its own shape, not by the
    side of the hand before it.
- A hand that appears, comes back after being lost, or changes side now starts clean on that side.
  Before, it could inherit the smoothing, trigger, grip and gesture of whatever that side held
  before, so a grip or gesture could survive a lost hand. A fast move alone doesn't reset anything,
  so a held grip survives a quick swing.
- In POV with the palm toward the camera, **Rebuild hands in 3D** could stay stuck with the index
  curled after the finger opened, so an open hand pulled the trigger to about half. The rebuild
  now also tries a start from MediaPipe's finger bends when its own fit disagrees with them by a
  lot. Not fully fixed: a blurred, fast-moving open hand can still give a short pinch spike.
- The driver could crash SteamVR when it shut down while the app was connected.
- The driver used to stop listening for good after one failed connection. Now it keeps listening.
- Hand position and rotation could be out of step for a frame.
- The driver log has fewer repeated lines: "camera direction set" and connection messages are
  limited.
- A malformed message to the SteamVR driver (a bad number, NaN, a huge value) could crash SteamVR.
  The driver now checks every message and ignores broken ones; trigger, grip and finger values are
  kept between 0 and 1.
- One bad tracking value (NaN or infinite) could freeze a hand until it left the camera's view: it
  stuck in the smoothing filters, and SteamVR dropped every message that carried it, trigger and
  grip included. The filters now ignore such a value, a hand with a pose SteamVR would reject is
  skipped for that frame (it keeps its last pose), and bad finger, trigger and grip values are sent
  as 0.
- Room-anchored hands kept following the head, and hand poses glitched now and then, when a message
  from the app reached the driver in two pieces.
- Holding R started a new recenter countdown on every key repeat; now it starts one.

## [1.4] - 2026-10-03

### Added
- **Rebuild hands in 3D** (Depth): fits a real hand model to the 2D points MediaPipe tracks well,
  so hands seen from behind (POV) turn suddenly much less often, keep a steadier distance, and
  fists are recognised. On in the POV preset.
- New built-in preset **POV Pointer**: POV with the hands turned for pointing.

### Changed
- **Facing** is the default preset. POV is retuned for the rebuilt hands (hand rotation, smoothing,
  hands slightly higher) and is still marked experimental.
- Presets you changed yourself are kept when you update.
- Settings added in a new version start with your active preset's value, so updating doesn't show
  "unsaved changes".

## [1.3] - 2026-10-02

### Added
- **Desktop app**: one window with the camera preview, a 3D view of where each hand is placed, a
  live readout of every finger, trigger and grip, and all settings. Most settings apply instantly.
- **Add-ons window**: checks the SteamVR driver at start and installs, updates or turns it back on
  with one button. The optional WiLoR depth installs from the same window.
- **Valve Index controllers** (optional, *Show hands as* → Index): games that support Index finger
  tracking show each of your fingers. Touch stays the default.
- **Presets per camera position**: POV (camera on your head or chest) and Facing (camera in front
  of you) are built in, and you can make your own.
- **Steady hand size** for steadier depth with a head or chest camera.

### Changed
- Trigger and grip are analog, from how far your fingers curl and pinch, instead of on/off
  gestures. Gesture detection is steadier.

### Fixed
- Left and right hands are correct with mirrored cameras.

### Removed
- The install scripts: the app installs the driver itself.

## [1.2] - 2026-10-02

### Added
- Optional **3D depth** (experimental, NVIDIA GPU): much steadier hand distance using the WiLoR hand
  model. It isn't bundled; an installer downloads it from the official sources. Personal,
  non-commercial use only, because of its licenses.

## [1.1] - 2026-10-02

### Fixed
- The tracker froze for about 2 seconds every few seconds when it couldn't reach the SteamVR driver.
  It now retries the connection in the background.

## [1.0] - 2026-10-02

### Added
- First ready-to-run Windows build: the SteamVR driver and the hand tracker, no Python or compiler
  needed.

[Unreleased]: https://github.com/Gann4Life/HandCameraDriverbyDmonOw/compare/v1.4...HEAD
[1.4]: https://github.com/Gann4Life/HandCameraDriverbyDmonOw/compare/v1.3...v1.4
[1.3]: https://github.com/Gann4Life/HandCameraDriverbyDmonOw/compare/v1.2...v1.3
[1.2]: https://github.com/Gann4Life/HandCameraDriverbyDmonOw/compare/v1.1...v1.2
[1.1]: https://github.com/Gann4Life/HandCameraDriverbyDmonOw/compare/v1.0...v1.1
[1.0]: https://github.com/Gann4Life/HandCameraDriverbyDmonOw/releases/tag/v1.0
