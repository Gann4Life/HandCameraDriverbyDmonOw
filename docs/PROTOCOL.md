# Tracker ↔ driver protocol

The contract between the Python tracker and the SteamVR driver. A change to a field changes both
sides and this document in the same commit.

| Side | Writes / reads it |
|---|---|
| Tracker | `HandData.to_protocol_string` and `protocol_greeting` (`hand_data.py`), sent by `utils/socket_client.py`; `RECENTER` from `HandTracker.step` (`Camera.py`) |
| Driver | `ParseGreeting`, `ParseHandMessage` and `LineSplitter` (`SteamVR Driver/src/hand_message.cpp`); the socket and greeting check in `TrackerServer` (`tracker_server.cpp`); messages applied by `HandTrackingListener::ProcessHandData` (`hand_tracking_listener.cpp`) |

## Transport

- TCP. The driver listens on `127.0.0.1:65432` (localhost only); the tracker connects as the client.
  The tracker's host and port come from `network.host` / `network.port` in the config; the driver's
  port is fixed.
- One client at a time. When it disconnects, the driver waits for the next one.
- UTF-8 text, one message per line, ended by `\n`. TCP may split or join lines: the driver buffers
  and only processes a line once its `\n` arrives. Empty lines are ignored, and a trailing `\r` is
  stripped. A line longer than 1024 bytes is dropped whole, up to its `\n`.
- The tracker reconnects on its own (every `reconnect_interval`, off the tracking thread).
- On Windows no other process can share the driver's port: if it is taken, the driver can't listen
  and logs it.

## Greeting

The first line the tracker sends on every connection, before any hand or recenter line:

```
HELLO:HANDCAM,VERSION:1,TYPE:INDEX
```

| Key | Value | Meaning |
|---|---|---|
| `HELLO` | `HANDCAM` | Marks the line as a greeting. The line must start with `HELLO:HANDCAM` |
| `VERSION` | integer, 1 or more | The protocol version the tracker speaks (see Versioning) |
| `TYPE` | `TOUCH` / `INDEX` | The controller profile, repeated from the hand lines (see below) |

Anything can connect to a local port, so the driver serves a connection only after a valid greeting.
It closes the connection when:

- no greeting arrives within 1 s of connecting (a slow drip of bytes doesn't extend this);
- the first line isn't a greeting: another program, an HTTP request, or a tracker older than this
  driver, which doesn't greet;
- `VERSION` differs from the driver's own;
- after the greeting, no complete line arrives for 5 s.

Each refusal reason is logged once, not on every retry. A version mismatch says which side to update.
Per-connection log lines (connected, disconnected, went silent, receive error) are limited to one a
second. After a refusal the driver goes on listening for the next client.

### Keepalive

The tracker sends nothing when it has no hand to report, and the driver closes a silent connection
after 5 s. So when the tracker has sent nothing for 1 s, it repeats the greeting line
(`SocketClient.keepalive()`, called each frame). After the greeting, the driver ignores a repeated
greeting; it doesn't treat it as a hand line. Drivers older than the greeting drop it too (no `HAND`).

`TYPE` is there for drivers older than the greeting. They read this line as a hand message (it has no
`HAND`, so they drop it) and fix the controller type from the first line they see, so the greeting
makes them pick the right type. The current driver takes the type from hand lines as before.

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
value are clamped to 0..1, and the quaternion is normalised. The tracker clamps `TRIGGER`, `GRIP`
and each `CURL` value before sending too, and sends `0` for any of them that is NaN or infinite.
It never sends a hand whose pose the driver would drop (NaN or infinite, beyond ±10 m, or a zero
quaternion): it skips that hand for the frame, so the driver keeps the controller's last state.

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

The protocol version is in the greeting: `PROTOCOL_VERSION` in `hand_data.py` and `kProtocolVersion`
in `hand_message.h`. The current version is 1. The driver refuses any version other than its own, so
`VERSION` counts breaking changes only. Adding an optional key must not raise it: the driver ignores
keys it doesn't know, so a newer tracker still works with it. Removing a key or changing its meaning
is a breaking change: bump the version on both sides in the same release.

Compatibility:

- A new tracker with an older driver (one that predates the greeting) works: that driver drops the
  greeting line and uses its `TYPE`.
- An old tracker with a new driver is refused and logged, because it doesn't greet.
- The tracker and the driver ship together, and the Add-ons window updates the driver.

## Known gaps

- The greeting is not authentication. Any local process can greet and send input. A per-session
  token may come later.
- `GESTURE` reaches the driver but drives nothing (buttons and stick are held at rest).
- The Index placement is corrected on both sides (see "Pose with `TYPE:INDEX`"); one side should own it.
- Switching away from `ANCHOR:ROOM` and back loses the last recenter.
