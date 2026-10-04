# Tracker ↔ driver protocol

The contract between the Python tracker and the SteamVR driver. A change to a field changes both
sides and this document in the same commit.

| Side | Writes / reads it |
|---|---|
| Tracker | `HandData.to_protocol_string` (`hand_data.py`), sent by `utils/socket_client.py`; `RECENTER` from `HandTracker.step` (`Camera.py`) |
| Driver | `HandTrackingListener::ProcessHandData` (`SteamVR Driver/src/hand_tracking_listener.cpp`) |

## Transport

- TCP. The driver listens on `127.0.0.1:65432` (localhost only); the tracker connects as the client.
  The tracker's host and port come from `network.host` / `network.port` in the config; the driver's
  port is fixed.
- One client at a time. When it disconnects, the driver waits for the next one.
- UTF-8 text, one message per line, ended by `\n`. TCP may split or join lines: the driver buffers
  and only processes a line once its `\n` arrives. Empty lines are ignored.
- The tracker reconnects on its own (every `reconnect_interval`, off the tracking thread).

## Message: hand

One per detected hand per frame:

```
HAND:LEFT,X:0.1200,Y:-0.3000,Z:-0.4000,QW:1.0000,QX:0.0000,QY:0.0000,QZ:0.0000,TRIGGER:0.80,GRIP:0.00,GESTURE:POINT,TYPE:INDEX,CURL:0.10;0.20;0.30;0.40;0.50,ANCHOR:ROOM
```

Comma-separated `KEY:VALUE` pairs. Order doesn't matter to the driver; unknown keys are ignored.

| Key | Value | Required | Meaning |
|---|---|---|---|
| `HAND` | `LEFT` / `RIGHT` | yes | Which controller. Any other value drops the line |
| `X`, `Y`, `Z` | float, metres, 4 decimals | yes | Hand offset in OpenVR axes (x right, y up, -z forward), added to the headset's position and rotated by the headset's orientation, or by the room forward when `ANCHOR:ROOM` |
| `QW`, `QX`, `QY`, `QZ` | float, unit quaternion | yes | Hand rotation, in the same frame as the position |
| `TRIGGER` | 0..1, 2 decimals | yes | Analog trigger. Touch above 0.1; click with hysteresis 0.95 on / 0.85 off |
| `GRIP` | 0..1, 2 decimals | yes | Analog grip. Touch above 0.1; force above 0.7 |
| `GESTURE` | `OPEN`, `FIST`, `POINT`, `PINCH`, `THUMBS_UP`, `PEACE`, `UNKNOWN` | yes | Recognised gesture. **Parsed but ignored by the driver** |
| `TYPE` | `TOUCH` / `INDEX` | no, default `TOUCH` | Controller profile. Taken from the first message; changing it needs a SteamVR restart |
| `CURL` | 5 floats 0..1, `;`-separated, thumb..pinky | no | Finger curls for the Index skeleton. Sent only when features exist |
| `ANCHOR` | `ROOM` | no | Camera fixed in the room: hands keep the room's forward instead of turning with the headset |

## Message: recenter

```
RECENTER:1
```

On a line of its own. The driver takes the headset's current facing as the room forward (used with
`ANCHOR:ROOM`). Until the first recenter, it takes the facing at the first `ANCHOR:ROOM` hand.

## Versioning

Optional keys are added at the end, and old drivers ignore what they don't know, so a newer tracker
works with an older driver (without the new feature). Removing a key or changing its meaning is a
breaking change: it needs a `VERSION` key first, and both sides updated in the same release.

## Known gaps

- The driver doesn't validate values: `std::stof` throws on a malformed number, and nothing catches
  it on the listener thread, which can take SteamVR's server down. NaN, infinities and out-of-range
  values aren't clamped either.
- `GESTURE` reaches the driver but drives nothing (buttons and stick are held at rest).
- The protocol has no version key.
