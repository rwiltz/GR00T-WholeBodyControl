# SONIC teleop control card

Everything is on the controllers. No keyboard.

| input | action |
|---|---|
| **left X** | switch mode: tracking ⇄ walking |
| **right A** | start / stop teleoperation |
| **right B** | reset the episode |
| **trigger** | pinch — index and thumb |
| **squeeze** | grasp — middle and thumb |
| **left stick** | tracking: slide yourself across the floor · walking: walk the robot |
| **right stick** ←→ | turn — the robot when walking, you when tracking |
| **right stick** ↑↓ | walking: raise / lower the hips |

**The floor tells you the mode.** Visible = walking. Hidden = tracking.

The left stick drives *relative to where the robot faces*, in eight 45° directions — so turn first,
then walk, exactly as on the real robot's gamepad.

Left hand drives the robot, right hand drives the session. Face buttons only, because
trigger and squeeze are the hands.

## Two things that surprise people

**Your view jumps when you enter walking mode.** It snaps to the robot; how far depends on how
much you have drifted apart. Leaving walking mode freezes it where it is.

**Reset is a single click.** It restores the episode and your viewpoint with no confirmation.

The console prints every mode switch and button press, so if something feels unresponsive, check
there before assuming the robot is at fault.

For setup, the task ids and what to expect from each mode, see
[the operator's guide](sonic_teleop_operator_guide.md).
