# Tracker ↔ driver protocol

The contract between the Python tracker and the SteamVR driver. A change to a field changes both
sides and this document in the same commit.

| Side | Writes / reads it |
|---|---|
| Tracker | `HandData.to_protocol_string` (`hand_data.py`), sent by `utils/socket_client.py`; `RECENTER` from `HandTracker.step` (`Camera.py`) |
| Driver | `ParseHandMessage` and `LineSplitter` (`SteamVR Driver/src/hand_message.cpp`), applied by `HandTrackingListener::ProcessHandData` (`hand_tracking_listener.cpp`) |

## Transport

- TCP. The driver listens on `127.0.0.1:65432` (localhost only); the tracker connects as the client.
  The tracker's host and port come from `network.host` / `network.port` in the config; the driver's
  port is fixed.
- One client at a time. When it disconnects, the driver waits for the next one.
- UTF-8 text, one message per line, ended by `\n`. TCP may split or join lines: the driver buffers
  and only processes a line once its `\n` arrives. Empty lines are ignored, and a trailing `\r` is
  stripped. A line longer than 1024 bytes is dropped whole, up to its `\n`.
- The tracker reconnects on its own (every `reconnect_interval`, off the tracking thread).

## Message: hand

One per detected hand per frame:

```
HAND:LEFT,X:0.1200,Y:-0.3000,Z:-0.4000,QW:1.0000,QX:0.0000,QY:0.0000,QZ:0.0000,TRIGGER:0.80,GRIP:0.00,GESTURE:POINT,TYPE:INDEX,CURL:0.10;0.20;0.30;0.40;0.50,ANCHOR:ROOM
```

Comma-separated `KEY:VALUE` pairs. Order doesn't matter to the driver; unknown keys are ignored.
Apart from `HAND`, the driver applies whichever fields arrive: a missing field keeps that controller's
previous value, and the line is not dropped.

The driver validates the whole line before applying any of it, and drops the line (logging it once)
when any field is malformed:

- a value that isn't a decimal number (empty, text, spaces, `nan`, `inf`, or out of float range);
- part of a group: `X` without `Y` and `Z`, or some of `QW`..`QZ` without the rest;
- a position coordinate beyond ±10 m, or a quaternion of length zero;
- a `CURL` without exactly five values;
- a `HAND` other than `LEFT` / `RIGHT`.

Values that are well-formed but out of range are fixed instead: `TRIGGER`, `GRIP` and each `CURL`
value are clamped to 0..1, and the quaternion is normalised. The tracker clamps `TRIGGER` and `GRIP`
before sending too.

| Key | Value | Sent | Meaning |
|---|---|---|---|
| `HAND` | `LEFT` / `RIGHT` | always | Which controller. Any other value drops the whole line |
| `X`, `Y`, `Z` | float, metres, 4 decimals | always | Hand offset in OpenVR axes (x right, y up, -z forward), added to the headset's position and rotated by the headset's orientation, or by the room forward when `ANCHOR:ROOM` |
| `QW`, `QX`, `QY`, `QZ` | float, unit quaternion | always | Hand rotation, in the same frame as the position |
| `TRIGGER` | 0..1, 2 decimals | always | Analog trigger. Touch above 0.1; click with hysteresis 0.95 on / 0.85 off |
| `GRIP` | 0..1, 2 decimals | always | Analog grip. Touch above 0.1; force above 0.7 |
| `GESTURE` | `OPEN`, `FIST`, `POINT`, `PINCH`, `THUMBS_UP`, `PEACE`, `UNKNOWN` | always | Recognised gesture. **Parsed but ignored by the driver** |
| `TYPE` | `TOUCH` / `INDEX` | when set; driver default `TOUCH` | Controller profile. Taken from the first valid line that isn't a recenter; changing it needs a SteamVR restart |
| `CURL` | 5 floats 0..1, `;`-separated, thumb..pinky | when features exist | Finger curls for the Index skeleton |
| `ANCHOR` | `ROOM` | when the camera is fixed | Camera fixed in the room: hands keep the room's forward instead of turning with the headset |

### Pose with `TYPE:INDEX`

The position and rotation always describe where a **Touch** controller would sit in the hand. With
`TYPE:INDEX` the driver moves that pose to an Index controller's origin with a fixed transform
(`controller_device_driver.cpp`, from the two render models' hand poses). The tracker's
`network.index_offset` / `network.index_rotation_deg` adjustment is applied before sending, on top
of that transform, so its tuned values assume the driver's correction.

## Message: recenter

```
RECENTER:1
```

On a line of its own. The driver takes the headset's current facing as the room forward (used with
`ANCHOR:ROOM`). Until the first recenter, it takes the facing at the first `ANCHOR:ROOM` hand.

Any hand line without `ANCHOR:ROOM` makes the driver forget the room forward, so after switching back
to a fixed camera the next `ANCHOR:ROOM` hand captures it again and the last recenter is lost.

## Versioning

Optional keys are added at the end, and old drivers ignore what they don't know, so a newer tracker
works with an older driver (without the new feature). Removing a key or changing its meaning is a
breaking change: it needs a `VERSION` key first, and both sides updated in the same release.

## Known gaps

- `GESTURE` reaches the driver but drives nothing (buttons and stick are held at rest).
- The protocol has no version key.
- The Index placement is corrected on both sides (see "Pose with `TYPE:INDEX`"); one side should own it.
- Switching away from `ANCHOR:ROOM` and back loses the last recenter.
