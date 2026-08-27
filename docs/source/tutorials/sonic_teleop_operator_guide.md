# SONIC teleoperation: operator's guide

How to drive the G1 in Isaac Lab from a headset. For setup and installation see
[the teleop tutorial](isaaclab_teleop.md).

## Start a session

```bash
python -m gear_sonic.lab_teleop.scripts.run teleop \
    --task IsaacContrib-Teleop-Sonic-WholeBody-G1-v0 \
    --viz kit --xr --device cuda:0
```

Connect the headset once the console prints `CloudXR runtime auto-launched`. Four tasks exist;
they differ only by SONIC checkpoint and by live-versus-replay:

| task id | |
|---|---|
| `IsaacContrib-Teleop-Sonic-WholeBody-G1-v0` | live, SONIC v1.1 |
| `IsaacContrib-Teleop-Sonic-WholeBody-G1-Replay-v0` | MCAP replay, v1.1 |
| `IsaacContrib-Teleop-Sonic-WholeBody-G1-LowLatency-v0` | live, low-latency checkpoint |
| `IsaacContrib-Teleop-Sonic-WholeBody-G1-LowLatency-Replay-v0` | MCAP replay, low-latency |

The low-latency checkpoint trails you by ~80 ms instead of ~200 ms. Fetch it with
`python download_from_hf.py --low-latency`.

## Controls

| input | action |
|---|---|
| **left** primary click | switch mode: tracking ⇄ walking |
| **right** primary click | show/hide the ground plane |
| trigger | pinch — index and thumb |
| squeeze | grasp — middle and thumb |
| left stick | **tracking mode**: slide yourself across the floor<br>**walking mode**: walk the robot |
| right stick, up/down | walking mode: raise/lower the robot's hips |
| `B` / `P` / `R` (keyboard) | start-resume / pause / reset |

Both toggles are clicks because trigger and squeeze drive the hands.

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
[SONIC] building the velocity planner (first entry to teleop mode)
[SONIC] ground plane shown
[SONIC] reset: XR anchor restored to (0.0, 0.0, -0.19)
```

If you press the left button and no `[SONIC] mode` line appears, the button is not reaching the
pipeline — a different problem from the mode behaving badly.

## What to expect

**Your viewpoint jumps when you enter walking mode.** It snaps to the robot, and how far it jumps
depends on how far you have drifted apart. Leaving walking mode freezes it where it is.

**Your wrists do not rotate in walking mode.** Arms reach, but hand orientation is not yet sent;
see [known gaps](#known-gaps).

**A hitch about once a second while walking** is the planner re-planning. It costs ~50 ms in a
single control step.

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

* **Hand rotation is not sent in walking mode.** The 83-wide reference carries only a root
  quaternion, so head and hand orientations are not recoverable from it.
* **The robot can fall in walking mode**, and the cause is not yet established. The zeroed hand
  orientations above are the leading suspect. Note whether it starts before or after your first
  mode switch -- that distinguishes the two paths.
* **Fingers render but only recently became actuated.** Grasping is new and lightly tested.
* **The props cost roughly 7.7 ms per frame**, taking replay from ~44 fps to ~33 fps.
