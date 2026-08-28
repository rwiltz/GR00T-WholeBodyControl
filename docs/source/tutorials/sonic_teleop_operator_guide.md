# SONIC teleoperation: operator's guide

How to drive the G1 in Isaac Lab from a headset. For setup and installation see
[the teleop tutorial](isaaclab_teleop.md).

## Start a session

```bash
python -m gear_sonic.lab_teleop.scripts.run teleop \
    --task IsaacContrib-Teleop-Sonic-WholeBody-G1-v0 \
    --viz kit --xr --device cuda:0
```

Connect the headset once the console prints `CloudXR runtime auto-launched`. Six tasks exist,
differing only by SONIC checkpoint, live-versus-replay, and whether the props are present. All are
prefixed `IsaacContrib-Teleop-Sonic-WholeBody`:

| task id | |
|---|---|
| `-G1-v0` | live, SONIC v1.1 |
| `-G1-Replay-v0` | MCAP replay, v1.1 |
| `-G1-LowLatency-v0` | live, low-latency checkpoint |
| `-G1-LowLatency-Replay-v0` | MCAP replay, low-latency |
| `-G1-LowLatency-Bare-v0` | live, low-latency, no props |
| `-G1-LowLatency-Bare-Replay-v0` | MCAP replay, low-latency, no props |

The low-latency checkpoint trails you by ~80 ms instead of ~200 ms. Fetch it with
`python download_from_hf.py --low-latency`.

## Controls

| input | action |
|---|---|
| **left** A/X (primary click) | switch mode: tracking ⇄ walking |
| **right** A (primary click) | start / stop teleoperation |
| **right** B (secondary click) | reset the episode |
| trigger | pinch — index and thumb |
| squeeze | grasp — middle and thumb |
| left stick | **tracking mode**: slide yourself across the floor<br>**walking mode**: walk the robot |
| right stick, left/right | **tracking mode**: turn yourself<br>**walking mode**: turn the robot |

Everything is on the face buttons because trigger and squeeze drive the hands. The left hand owns
the robot's mode; the right hand owns the session. No keyboard is needed.

Right stick up/down is unused: hip height went away with the port to the robot's own stick
semantics, which send "use the clip default" for height. See
[the control card](sonic_teleop_controls.md) for the one-page version.

**The floor tells you which mode you are in** — visible in walking mode, hidden in tracking mode.
It is not a separate control.

## The two modes

You start in **tracking** mode after every reset.

**Tracking** — the robot copies your whole body. Your own displacement drives its gait, so walking
in the room walks the robot. The left stick slides your viewpoint across the floor without moving
the robot, which is how you reach the props without a large play space.

**Walking** — the left stick drives the robot's gait through a motion planner, and your viewpoint
rides along with the robot. Use this to cover ground.

Watch the console; it prints only on change:

```
[SONIC] mode smpl -> teleop
[SONIC] ground plane shown
[SONIC] teleop started (right A)
[SONIC] reset (right B)
[SONIC] reset: XR anchor restored to (0.0, 0.0, -0.19)
```

If you press the left button and no `[SONIC] mode` line appears, the button is not reaching the
pipeline — a different problem from the mode behaving badly.

## What to expect

**Your viewpoint jumps when you enter walking mode.** It snaps to the robot, and how far it jumps
depends on how far you have drifted apart. Leaving walking mode freezes it where it is.

**A hitch about once a second while walking** is the planner re-planning. It costs ~10 ms in a
single control step, inside the 20 ms budget.

**One visible step change at the moment you switch modes.** The encoder changes mode, so its
output moves in a single control step. Upstream cross-fades successive plans over 0.16 s; we do
not blend yet, because the sustained instability turned out to be elsewhere. If this single step
proves too abrupt in practice it is the next thing to add.

**Startup takes about two seconds longer than it used to.** The motion planner is now built and
warmed up before the first frame. It used to be built the first time you pressed into walking
mode, which stalled the control loop for ~1.7 s -- 87 steps -- at exactly the moment the robot
started walking.

**Sliding yourself around is smooth VR locomotion** and makes some people queasy. The speed is
`anchor_pan_speed` on the action config, 1.0 m/s by default; set it to `0.0` to disable.

**Nothing tethers you to the robot in tracking mode.** Slide far enough and it leaves view. Switch
to walking mode to snap back, or reset.

## The scene

A packing table sits five feet in front of the robot, carrying crates, boxes and a steering wheel.
The wheel is grabbable and will stay wherever it ends up — only a reset puts it back. The table
itself is fixed. Both stream from the Omniverse content bucket, so the first launch on a machine
pays a download and an offline machine will not spawn them.

## Known gaps

* **The robot can fall in walking mode**, and the cause is not yet established. Zeroed wrist and
  head orientations used to be the leading suspect and have now been fixed, so if falls persist
  the cause lies elsewhere. Note whether it starts before or after your first mode switch --
  that distinguishes the two paths.
* **Fingers render but only recently became actuated.** Grasping is new and lightly tested.
* **The props cost roughly 7.7 ms per frame**, taking replay from ~44 fps to ~33 fps.
