# SONIC teleop control card

Everything is on the controllers. No keyboard.

| input | tracking mode | walking mode |
|---|---|---|
| **left X** | switch to walking | switch to tracking |
| **right A** | start / stop teleoperation | ← same |
| **right B** | reset the episode | ← same |
| **left stick** | slide yourself across the floor | walk the robot |
| **right stick ←→** | turn yourself | turn the robot |
| **trigger** | pinch — index and thumb | ← same |
| **squeeze** | grasp — middle and thumb | ← same |

**The floor tells you the mode.** Visible = walking. Hidden = tracking.

Left hand drives the robot, right hand drives the session. The left stick moves you *relative to
where you face*, so turn first, then walk.

## Stick feel

Matched to the real robot's, from the same source it runs
(`pico_manager_thread_server.py:1795-1876`), so the two should feel the same:

* turn sweeps at **1.5 rad/s** (86°/s) at full deflection
* both sticks have a **0.15 deadzone**, and movement is rescaled past it so it ramps from zero
* movement is **continuous** — no snapping to fixed directions
* a centred stick drops to the idle gait rather than walking on the spot

## Two things that surprise people

**Your view jumps when you enter walking mode.** It snaps to the robot; how far depends on how
much you have drifted apart. Leaving walking mode freezes it where it is.

**Reset is a single click**, with no confirmation. It restores the episode and your viewpoint.

## Not wired up

* **Right stick ↑↓ does nothing.** Hip height went away with the port to the robot's semantics —
  it sends "use the clip default" for height, so there is no per-operator control of it.
* **Left Y is unused.**
* Fingers render and actuate, but grasping is lightly tested.

The console prints every mode switch and button press. If something feels unresponsive, look there
before assuming the robot is at fault.
