# How the hand tracking works

A study of the tracker for anyone curious: how one ordinary webcam becomes two VR controllers, what
each step does, why it is there and where it still falls short. [ARCHITECTURE.md](ARCHITECTURE.md)
covers how the code is organised; this document covers the algorithm. Names in `code` point to
where each step lives.

## The idea in one paragraph

Every camera frame goes through a chain of stages. A neural network (MediaPipe Hands) finds up to
two hands and their 21 joints in the picture. Everything after it is ordinary geometry and
filtering, and most of it is there to correct something the stage before gets wrong:

- the network's left/right labels flicker, so they are only one vote among several;
- the picture is flat, so the distance to the camera has to be inferred;
- each frame's answer is noisy, so it is smoothed;
- a fast hand blurs and is lost for a few frames, so it is predicted until it is found again.

The result is a position, a rotation, finger curls, a trigger and a grip per hand, sent about 30
times a second to a SteamVR driver that shows them as two controllers.

## The chain at a glance

```
camera frame
  │
  ├──────────────────────────────────────────────► optical flow (only while a hand is lost)
  ▼
① MediaPipe Hands ── 21 points in the picture, 21 points in metres, a left/right guess
  ▼
② Handedness vote ── this frame's left/right evidence, from the hand's shape
  ▼
③ Identity ───────── which detection is the left hand and which the right, with memory
  ▼                  (drops duplicates; a new hand on a side resets that side's state)
④ 3D hand ────────── the hand's shape in metres (MediaPipe's, or rebuilt from the 2D points)
  ▼
⑤ Distance ───────── where the wrist is in front of the camera (perspective fit)
  ▼
⑥ Rotation ───────── which way the palm and fingers point
  ▼
⑦ Smoothing ──────── One Euro filters: steady at rest, quick in motion
  ▼
⑧ Motion memory ──── records the path; blends a returning hand back in
  ▼
⑨ Headset space ──── camera space to where the hand is relative to your head (or the room)
  ▼
⑩ Fingers ────────── curls, pinch, trigger, grip, gesture name
  ▼
⑪ Lost hands ─────── a hand not found this frame carries on along its path (+ optical flow)
  ▼
⑫ Driver ─────────── one text line per hand over a local socket; SteamVR shows the controllers
```

So yes, mostly a chain of correction layers, with two caveats. First, not every stage corrects:
② to ⑥ and ⑨ to ⑩ convert one representation into another, and their errors pass straight on.
Second, the chain is not always the same length: ⑪ only runs for a hand that ① did not find, and
it reads the memory that ⑧ kept from earlier frames instead of anything from this frame.

Each stage keeps its state per hand, keyed by side ("left", "right"), and both hands go through the
same code.

## ① Detection: MediaPipe Hands

`HandTracker.step` in `Camera.py` hands the newest frame to Google's MediaPipe Hands. Frames are
never queued: if tracking falls behind, older frames are skipped, because a late answer is worse
than a missing one. For each hand it finds, MediaPipe returns:

- **21 image landmarks**: the wrist and four joints per finger, as positions in the picture (0 to 1
  across and down), plus a relative depth that is not in metres and is not used for placement.
- **21 world landmarks**: the same joints in metres, centred on the hand. This is MediaPipe's guess
  at the hand's 3D shape, but not at where the hand is.
- **A handedness label** ("Left" or "Right") with a confidence score.

MediaPipe follows a hand from frame to frame by itself, and runs its slower palm detector only when
it loses one. That is where most of the problems below come from: on a fast move the hand blurs, the
tracking fails, and the detector needs a few frames to find it again.

Its 2D points are the most reliable thing in the whole chain. Its 3D shape is good when the palm
faces the camera and poor from behind (see ④). Its handedness label is the weakest: it assumes a
selfie view and flips when the hand turns.

## ② Handedness vote

`HandTracker.handedness_evidence`, `gesture_detector.py`, `combine_handedness_votes` in
`hand_identity.py`.

Each detection gets a vote between −1 (right) and +1 (left). It is only a vote: ③ decides. The vote
mixes two cues:

- **Finger curl (80 %).** Fingers only bend toward the palm. The side the fingertips bend to tells
  the palm side, and from that the hand's chirality, whatever the hand's rotation. It is strong for a
  curled hand and says little about a flat one.
- **A secondary cue (20 %).** In POV (camera on your head, seeing the backs of your hands) it is the
  geometry of a hand whose back faces the camera. Otherwise it is MediaPipe's own label.

The curl cue reads MediaPipe's 3D shape. When MediaPipe sees the back of the hand, that shape can be
inside out, which reverses the curl vote. In POV the secondary cue uses only the picture's x and y,
so it cannot be fooled that way. When the two clearly disagree, the frame casts no vote.

A mirrored camera (some phone-camera apps mirror the picture) shows a right hand as a left one, so
every image-based cue is flipped back for a mirrored frame.

## ③ Identity: which hand is which

`HandIdentityTracker` in `hand_identity.py`.

Deciding left and right from scratch every frame makes hands swap whenever the vote flickers. Here
the identity follows **continuity first**, and the vote only overrides it when it persists.

1. **Duplicates.** MediaPipe sometimes reports one hand twice. Detections with wrists closer than
   `duplicate_radius` (5 % of the picture) are one hand, and the more confident copy is kept. At
   most two hands are kept.
2. **One hand in view.** It keeps the side of the remembered hand whose wrist is within
   `continuity_radius` (15 % of the picture) of it. A lone hand keeps its side even when it jumps
   further, as with a turning head camera, unless a strong vote says otherwise after a break. A
   continuing hand changes side only after `switch_frames` frames in a row of strong contrary votes
   (about half a second).
3. **Two hands in view.** A hand that continues a remembered one keeps its side, and the newcomer
   takes the other. Otherwise both ways of assigning them are scored on three things: distance to
   the remembered hands, the votes, and left-right order (with the hands clearly apart, the left
   hand is on your left). The cheaper assignment wins. The order term lets a swapped pair correct
   itself, since continuity alone would keep the swap for ever. The price is that crossed arms held
   apart get swapped.
4. **Memory.** A lost hand is remembered for `memory_seconds`, or as long as it is predicted (⑪) if
   that is longer, so a hand found again where it was predicted is the same hand.
5. **New on a side.** A hand that just appeared, came back after its memory ran out, or changed side
   is "new". The tracker then resets that side's filters, fit, grip, gesture and motion memory, so
   nothing from a different hand or moment leaks in. A fast move alone is not new: resetting a held
   grip mid-swing would drop what the hand holds.

## ④ The hand's 3D shape

Placement and rotation both need the hand's shape in metres. There are two sources.

**MediaPipe's world landmarks (Facing, the default).** Good when the palm or the side of the hand
faces the camera.

**The rebuilt hand (POV, `calibration.rebuild_hand`, `hand_fit.py`).** Seen from behind, MediaPipe
places the 2D points well but gets the 3D shape wrong: knuckle depths flip and the hand folds onto
itself. The rebuild fits a hand model to the 2D points alone. The model has 26 parameters:
rotation, position, 4 thumb angles and 4 angles per finger. It has MediaPipe's average proportions,
and its joints only bend the ways a real hand can, so it can only produce shapes a real hand can
make.

- The fit is a Levenberg-Marquardt least-squares solve that minimises how far the model's projected
  joints land from MediaPipe's 2D points, measured in pixels.
- Soft terms keep it plausible: a fingertip joint follows its middle joint, the pose stays near last
  frame's, and the joint bends stay loosely near the ones MediaPipe read. MediaPipe gets each
  finger's bend about right even when the hand's overall shape is wrong, and the bends settle what
  the 2D points can't, such as a finger pointing straight at the camera.
- Each frame starts from last frame's fit (4 iterations). When that fits badly (over 8 px RMS), or
  its bends disagree a lot with MediaPipe's, it starts over from fresh poses: an open hand, a fist
  and MediaPipe's bends, each placed every way the palm fits. They are scouted briefly and the best
  two are finished. Starting over is slow, so it happens at most every 0.2 s.
- In POV, when two poses fit about equally well, the one with the back of the hand toward the camera
  wins.

From here on the rebuilt hand stands in for MediaPipe's: distance, rotation and finger curls are all
read from it.

## ⑤ Distance from a single camera

`HandTracker.fit_hand`, `utils/hand_size.py`.

One camera can't see depth. What it can see is size: a hand twice as far away looks half as big. So
the distance comes from fitting a hand of known size into the picture. OpenCV's `solvePnP` (the
SQPnP solver) finds the rotation and position that make the 3D hand, from ④, project onto the 21
image points. The camera's focal length follows from its horizontal field of view (`camera.hfov_deg`,
70° by default).

This gives the wrist's position in metres, in camera space. Across the picture it is accurate,
because it is pinned to the 2D points. Along the camera's axis it is the weakest number in the
whole tracker:

- **Size error becomes distance error.** If the hand model is 1 % too big or small, the distance is
  1 % off: 5 mm at 50 cm. MediaPipe's 3D hand changes size by a few percent from frame to frame,
  which shows up as the hand moving toward and away from the camera by centimetres.
- **Shape error becomes distance error.** A finger hiding behind the palm, or the wrist turning
  edge-on, changes the apparent size of the hand without the hand moving.
- **Your hand isn't the average hand.** `calibration.hand_scale` sizes the model to your hand. Without
  it, distances are right relative to each other but scaled.

Two things help. **Steady hand size** (`calibration.steady_hand_size`, on in POV) rescales each
frame's model to the median size of that hand's last 90 frames: a real hand doesn't change size, so
the wobble is removed. Its rigid palm edges are measured, not the fingers. The **depth filter** (⑦)
then smooths depth harder than the other axes.

If the fit fails, the hand keeps its last position. A hand without one yet is placed at a fixed
arm's length, 45 cm.

(`tracking.depth_source: wilor` is an experimental alternative that takes the distance from a
separate 3D hand model run on the GPU. It isn't bundled, is for non-commercial use only, and is
planned for removal.)

## ⑥ Rotation

`GestureDetector.calculate_hand_orientation` in `gesture_detector.py`.

The rotation is built from three points of the 3D hand:

- **Forward:** from the wrist to the middle of the four knuckles. This is the direction the fingers
  point.
- **Across:** from the little finger's knuckle to the index knuckle.
- **Back of the hand:** perpendicular to both. Which way it points depends on the hand: the thumb is
  on opposite sides of each hand, so the order of the cross product is swapped. This is why ③ has
  to be right before the rotation can be.

The three axes form the rotation in OpenVR's controller convention: −z along the fingers, +y out
of the back of the hand. A flat hand, palm down, fingers away from you, is no rotation at all. It
is then tilted by how the camera is mounted (`calibration.camera_rotation_deg`) and turned by a
per-hand offset (`calibration.rotation_offset_deg`), so the virtual controller sits in the hand the
way a real one would.

## ⑦ Smoothing

`HandTracker.build_filters`, `utils/one_euro.py`.

Each frame's measurement jitters. A plain average removes jitter but makes the hand lag behind
every move. The **One Euro filter** (Casiez, Roussel and Vogel, CHI 2012) adapts:

```
speed   = smoothed rate of change of the signal   (smoothed at d_cutoff Hz)
cutoff  = min_cutoff + beta × speed
output  = low-pass of the signal at that cutoff
```

At rest the cutoff is low, so the output is steady. Moving fast, the cutoff rises, so the output
follows closely. Jitter is visible when a hand is still and lag is visible when it moves, so each
is fought where it shows.

Each hand gets three filters:

| Filter | Rest cutoff | `beta` | Why separate |
|---|---|---|---|
| Position across the picture (x, y) | 1 Hz | 1.5 | Accurate, so lightly smoothed |
| Depth (z) | 0.3 Hz | 2 | Far noisier (see ⑤), so smoothed much harder at rest |
| Rotation (quaternion) | 1 Hz | 0.5 | Smoothed on the sphere of rotations; q and −q are kept on the same side so it never averages across the flip |

The x and y filter treats the position as one point, with one speed, not as two unrelated
numbers. A filter that hears nothing for 0.5 s starts over, so a hand that reappears jumps to its
place instead of gliding in from where it disappeared. A NaN or infinite value is ignored instead
of being kept, since a kept one would freeze the hand. The POV presets use a simpler exponential
filter (EMA, a fixed fraction of each new frame) instead. **none** turns smoothing off, for
comparison.

Smoothing has a price: while a hand moves, the filtered depth trails the measured one by a few
centimetres, most visibly on a punch toward or away from the camera.

## ⑧ Motion memory

`MotionPredictor.observe` in `hand_motion.py`.

Every tracked frame's smoothed pose is recorded: the newest 8 frames within 0.15 s. That is what
⑪ predicts from. When a hand comes back after being predicted, its measured pose usually differs
from the predicted one. Jumping there would look like a glitch, so the difference is faded out over
0.1 s (a smoothstep, which has no kink at either end). The history keeps the measured frames, not
the blended ones, so errors don't build up over repeated dropouts.

## ⑨ From camera space to headset space

`HandTracker.driver_pose`, then the driver.

The camera's view is turned into where the hand is relative to you:

- **Facing** (camera in front of you, looking back): the depth axis is reversed, and the distance to
  the camera (`camera.facing_distance`) is taken off, so a hand nearer the camera is further out in
  front of you.
- `calibration.scale`, then the camera's mounting tilt, then `calibration.position_offset` (where the
  camera sits relative to your eyes).
- For Index controllers, a small extra offset and turn (`network.index_*`), because an Index
  controller sits in the hand differently from a Touch.

The driver adds the result to the headset's pose, so the hands move with your head, which is right
for a camera on your head. With **Hands follow: The room** (a camera standing in the room), the
driver uses a fixed room forward instead of the headset's facing, and **Recenter** sets that
forward. The driver does no smoothing or prediction of its own: it shows what it is sent.

## ⑩ Fingers, trigger, grip and gestures

`hand_features.py`, `hand_controls.py`, `gesture_scores.py`.

Everything here is scale-free, made of angles or of distances divided by the palm's size, so it
doesn't depend on how big the hand is or how far it is from the camera.

- **Curl per finger:** the sum of the bends at its joints, mapped from the open angle (0) to the
  fully curled angle (1). An open hand isn't at 0° (the wrist-to-knuckle line isn't in line with the
  finger), so both ends come from recordings, and **Calibrate gestures** replaces them with your own.
- **Pinch:** the distance from the thumb tip to each fingertip, in palm sizes: 0.55 is apart, 0.20 is
  touching. A thumb against a fist is not a pinch: the pinch fades out when the index tip is in the
  palm.
- **Trigger:** how far the index curls, or how hard thumb and index pinch, whichever is more.
- **Grip:** the average curl of middle, ring and little finger.
- Both are smoothed with a One Euro filter. Once the grip reaches **Grip holds from** it holds its
  peak until it stays below **Grip lets go below** for 0.1 s, so a grip near the game's threshold
  doesn't grab and drop. Both stay analog.
- **Gestures** (open, fist, point, pinch, thumbs up, peace) each get a 0-to-1 score. A gesture scores
  as well as its worst-matching finger, a fuzzy AND, so "open" needs every finger open. The name
  shown changes only with hysteresis: a score of 0.6 to enter, 0.4 to leave, a margin to take
  over, and 3 frames of confirmation.
- For Index controllers, each finger's curl is sent too, so games with finger tracking show each
  finger.

## ⑪ Lost hands: prediction and optical flow

`HandTracker.predict_lost_hands`, `hand_motion.py`, `hand_flow.py`.

On a fast move the hand blurs and MediaPipe loses it for a few frames. Left alone, the virtual hand
would freeze where it was last seen and jump when found. Instead, a lost hand keeps moving for up to
**Keep lost hands moving for** (0.3 s by default).

**Along a curve, from its last frames.** The heading and speed of each step between the tracked
frames are each fitted with a straight line against time. A swing therefore keeps bending as it
was, and an arc around the elbow keeps its radius. A hand that was slowing keeps slowing. Only the
steps since the hand last reversed are used, so a punch pulled back isn't read as a sharp turn. The
hand then:

- keeps its full speed for 0.05 s, then eases off with a 0.2 s time constant, so every prediction
  comes to rest;
- never speeds up past the speed it was lost at (a swing is fastest mid-arc, and speeding up
  overshot), and never goes past 4 m/s, beyond which the speed is a misread;
- keeps turning (its rotation's average rate over the history), up to 90° in all.

**Guards, each from a failure seen in recordings:**

- **Depth** carries on only when at least 3 tracked steps agree on it, all the same way and none
  making most of the move. One camera's depth jumps by centimetres when the wrist turns, and
  extrapolating such a jump sent hands 8 to 14 cm away.
- **A wrist that jumps** across the picture for one frame (a misdetection, or the other hand) is left
  out of the fit.
- **A hand that was barely moving** across the picture (under half a picture a second) holds still:
  its speed would be mostly noise. Steady depth still carries on, so a punch at the camera does too.
- **A hand about to leave the picture** eases to a stop at once instead of flying off.

**Optical flow.** The last tracked frames can't tell a swing that is just starting from a hand at
rest, but the picture can: the blurred hand is still in it. While a hand is lost, `HandFlow` runs
OpenCV's DIS optical flow between consecutive frames, on a quarter-size grey copy (about 1.5 ms),
inside the hand's last box, grown by 30 %. The flow of the picture outside the box is the
background, which moves as a whole when the head turns. The quarter of the box's pixels that move
most unlike it are taken as the hand. Their median shift moves the box along, and the hand follows
it across the picture. Depth and rotation still come from the curve fit. When the box leaves the
picture, the flow stops and its last offset holds.

A predicted hand keeps the fingers, trigger, grip and gesture of its last tracked frame. ③'s memory
of the lost hand is moved along with the prediction, so when MediaPipe finds the hand where it was
predicted, it is recognised as the same hand.

## ⑫ To SteamVR

`HandData.to_protocol_string` in `hand_data.py`, the driver in `SteamVR Driver/src`.

Each hand becomes one text line: position, rotation quaternion, trigger, grip, gesture, controller
type, finger curls and, for a camera fixed in the room, an anchor flag. The line goes over a local
socket to the driver, which validates every field and moves two virtual controllers (Touch or Index
profile). A hand with a pose SteamVR would reject (a NaN, an infinite value) is skipped for that
frame and keeps its last pose. The format is specified in [PROTOCOL.md](PROTOCOL.md).

## Known limits

Measured on recorded sessions (the app's **Record** saves the video plus what the tracker made of
every frame, and the same frames can be re-tracked offline to compare changes):

- **Depth is the weak axis.** It comes from apparent size, so it is centimetres noisier than the
  other two axes, and smoothing it adds lag while the hand moves.
- **Depth during a dropout can't be predicted.** Over about 200 lost-hand gaps, no fit of the frames
  before a gap (trends, robust slopes, the flow's change of scale) guessed where the depth went
  better than holding it. That is why depth only carries on when its frames clearly agree.
- **A lost hand just starting to move can pause.** Its last frames show it at rest, and the flow
  helps only while the blurred hand stays visible.
- **Crossed arms held apart are swapped** by the left-right order rule in ③.
- **In POV, a palm turned toward the camera** casts no handedness vote, so a hand on the wrong side
  can't correct itself while held that way.
- **One camera, one viewpoint.** A hand hidden behind the other, or edge-on to the camera, is
  guessed, not seen.

## Glossary

- **Landmark:** one of MediaPipe's 21 points per hand (the wrist, then four per finger).
- **Camera space:** metres from the camera; in this project's OpenVR axes, x right, y up, −z
  forward.
- **PnP (perspective-n-point):** finding where a known 3D object is from where its points appear in
  the picture.
- **Quaternion:** four numbers for a rotation, without the gimbal problems of three angles.
- **One Euro filter:** a smoothing filter whose strength drops as the signal speeds up.
- **Optical flow:** how far each pixel moved between two frames.
