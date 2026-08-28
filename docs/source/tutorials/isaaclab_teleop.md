# Isaac Lab Teleoperation and Data Collection

Drive a Unitree G1 in Isaac Lab with the SONIC whole-body controller, using Isaac Teleop full-body
XR tracking as the input. Every joint — legs, torso, arms, wrists — is produced by SONIC. There is
no IK and no scripted locomotion.

```{admonition} How this differs from the other Isaac Teleop page
:class: note
[Isaac Teleop Setup (CloudXR / DeviceIO, in-process)](isaac_teleop_publisher_setup.md) covers the
**deployment** path: `pico_manager_thread_server.py` hosts CloudXR, retargets, and publishes over
ZMQ to the C++ controller on the robot (or to MuJoCo).

This page covers the **simulation** path: Isaac Lab hosts the XR session itself, retargeting runs
in-process as an Isaac Teleop pipeline node, and SONIC runs inside an Isaac Lab `ActionTerm`. There
is no ZMQ, no separate publisher process, and no second virtual environment.

Both consume the same headset tracking data. Only one may own the OpenXR session at a time, so do
not run them simultaneously.
```

## What runs

```
Isaac Teleop (FullBodySource, 24-joint XR_BD_body_tracking)
  ├─ SonicFullBodyRetargeter        XR joint rotations -> SONIC reference (95 floats)
  ├─ SonicPicoLocomotion            thumbsticks -> planner command, the robot's own semantics
  └─ TriHandMotionController (x2)   trigger/squeeze -> finger joints
       └─ env.step(action)          standard gym contract, 121 floats
            └─ SonicModalWholeBodyAction
                 encoder -> 64-dim token -> decoder -> 29 raw actions
                 └─ apply_actions   raw * scale + default_joint_pos -> G1 joint targets
```

Both of SONIC's operator modes are driven, switched from the left controller. In `smpl`
(`mode_id 2`) the operator's legs drive the robot's legs. In `teleop` (`mode_id 1`) the upper body
still tracks while a kinematic planner supplies the lower body, so the sticks walk the robot; that
planner runs inside the action term, closed-loop on measured robot state.

## Prerequisites

1. **Isaac Lab 3.0** (`3.0.0-beta2` or newer), as a **source checkout** — the stock teleoperation
   scripts live under `scripts/` and are not shipped in the wheel, so a wheel-only install is not
   enough.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   uv run isaaclab --help     # first run creates .venv and installs dependencies
   ```

   This creates `IsaacLab/.venv`, which is the environment every command below runs in. Either
   activate it, or target it explicitly:

   ```bash
   source /path/to/IsaacLab/.venv/bin/activate
   # or prefix each install with:  uv pip install --python /path/to/IsaacLab/.venv/bin/python ...
   ```

   See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   for alternatives and troubleshooting.
2. **This repo**, installed into that same Isaac Lab environment (from the
   `GR00T-WholeBodyControl` root):

   ```bash
   pip install --no-deps -e ./gear_sonic
   ```

   ```{admonition} Install with --no-deps
   :class: warning
   `gear_sonic` pins `numpy==1.26.4` and `scipy==1.15.3` for Isaac Lab 2.3.2. Isaac Lab 3.0 ships
   numpy 2.x, so resolving those pins **downgrades numpy and scipy in Isaac Lab's environment**,
   which can break Isaac Sim. This workflow uses only `gear_sonic`'s pure rotation and SMPL
   forward-kinematics helpers, which run correctly on numpy 2.x. For the same reason this workflow
   has no `pip` extra — an extra would be resolved against those pins.
   ```

3. **The ONNX GPU runtime.** Pick the line matching the CUDA major version of the torch build in
   this environment:

   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

   | `torch.version.cuda` | install |
   |---|---|
   | `12.x` | `pip install "onnxruntime-gpu[cuda,cudnn]>=1.20,<1.27"` |
   | `13.x` | `pip install "onnxruntime-gpu[cuda,cudnn]>=1.27,<1.30"` |

   What matters is torch's CUDA build, **not** the host CUDA toolkit or Linux distribution: torch
   and onnxruntime both take their CUDA runtime from pip wheels. The `[cuda,cudnn]` extras pull
   those wheels explicitly, so nothing depends on a system CUDA install — and in an Isaac Lab
   environment they are usually already satisfied by torch, making them a no-op.

   Getting this wrong is quiet rather than loud. `onnxruntime-gpu` 1.28 against a `cu12` torch fails
   with `libcublasLt.so.13: cannot open shared object file`, and onnxruntime then falls back to CPU
   **silently** — where the SONIC decoder costs ~17 ms against a 20 ms control period. The action
   term refuses to start in that state rather than running slowly without saying so, and the error
   names the uninstall/install pair that fixes it. Verify ahead of time with
   `check_environment.py --lab-teleop` (below) rather than inferring from frame rate.

4. **SONIC checkpoints**:

   ```bash
   python download_from_hf.py --sonic-v1-1
   ```

5. **Git LFS**, for the G1 meshes: `git lfs pull`.

### Verify

```bash
python check_environment.py --lab-teleop
```

Checks Isaac Lab, `isaacteleop`, `gear_sonic`, the SONIC checkpoint, and that the CUDA execution
provider actually loads.

## Tasks

Six ids, all prefixed `IsaacContrib-Teleop-Sonic-WholeBody`:

| Task id | Checkpoint | Input | Props |
|---|---|---|---|
| `-G1-v0` | v1.1 | live XR headset | yes |
| `-G1-Replay-v0` | v1.1 | recorded MCAP | yes |
| `-G1-LowLatency-v0` | low-latency | live XR headset | yes |
| `-G1-LowLatency-Replay-v0` | low-latency | recorded MCAP | yes |
| `-G1-LowLatency-Bare-v0` | low-latency | live XR headset | no |
| `-G1-LowLatency-Bare-Replay-v0` | low-latency | recorded MCAP | no |

Live and replay are separate ids because `TeleopSession` rejects source nodes carrying a tracker
vendor when the session mode is `SessionMode.REPLAY`, so the replay tasks use a vendor-less
pipeline. `Bare` drops the packing table and its crates. The low-latency checkpoint trails the
operator by ~80 ms against v1.1's ~200 ms.

## Running

All entry points are Isaac Lab's own scripts, invoked through a thin launcher that registers this
repo's task ids first (see [Why a launcher](#why-a-launcher)). Everything after the script name is
forwarded to the stock script unchanged, so its `--help` and flags apply as documented upstream.

### Teleoperate

```bash
python -m gear_sonic.lab_teleop.scripts.run teleop \
    --task IsaacContrib-Teleop-Sonic-WholeBody-G1-v0 \
    --viz kit --xr
```

Wraps `scripts/environments/teleoperation/teleop_se3_agent.py`.

```{admonition} The SE(3) devices do not apply here
:class: warning
Do **not** pass `--teleop_device keyboard|spacemouse|gamepad`. Those emit a 6-DoF pose delta plus a
gripper value, which cannot express a whole-body pose — and passing the flag also forces the legacy
device path, bypassing the Isaac Teleop pipeline entirely. Leave it unset so the environment's
`isaac_teleop` config is used.
```

### Record demonstrations

```bash
python -m gear_sonic.lab_teleop.scripts.run record_demos \
    --task IsaacContrib-Teleop-Sonic-WholeBody-G1-v0 \
    --dataset_file ./datasets/sonic_g1_demos.hdf5 \
    --num_demos 10 \
    --viz kit --xr
```

Wraps `scripts/tools/record_demos.py`, which builds its own recorder manager and writes HDF5.

Add `--mcap_record_path ./captures/session.mcap` to also capture the raw teleop session. Those
captures include the `_teleop_control` channel carrying the operator's START/STOP gestures, which is
what makes them replayable (see below).

### Replay a capture

```bash
python -m gear_sonic.lab_teleop.scripts.run replay \
    --task IsaacContrib-Teleop-Sonic-WholeBody-G1-Replay-v0 \
    --replay_file ./captures/session.mcap \
    --viz kit
```

Wraps `scripts/environments/teleoperation/teleop_replay_agent.py`. No headset required, which makes
it the practical way to develop and debug the pipeline.

```{admonition} Captures recorded outside record_demos.py
:class: note
The replay agent gates `env.step()` on a START edge from the `_teleop_control` channel. A bare
tracker recording has no such channel, so nothing ever steps and the run reports `frames=0`. Isaac
Lab does not currently expose a flag to bypass this. Prefer captures produced by
`record_demos.py --mcap_record_path`.
```

The replay agent is a timing harness and requires rendering; run it with `--viz kit` or it aborts
with *"no usable frame intervals"*.

## Expected performance

Measured on a Threadripper 7960X / RTX PRO 6000, single environment:

| | rate |
|---|---|
| SONIC encoder + decoder (CUDA) | 0.74 ms |
| SONIC encoder + decoder (CPU) | 17.4 ms — fatal, see above |
| Environment, headless | ~66 Hz |
| Environment, `--viz kit` | ~33 Hz |

The control rate is 50 Hz (200 Hz physics, `decimation = 4`), matching SONIC's own `control_dt`.
The gap between headless and windowed is rendering, not inference.

### Why onnxruntime

SONIC is run through the released ONNX graphs, so simulation executes the same weights and graph as
the robot. The alternatives were considered:

- **TensorRT** is what the C++ deploy stack uses on the robot, and onnxruntime can reach it via
  `TensorrtExecutionProvider`. It is not the default here: it requires additional NVIDIA packages
  beyond `onnxruntime-gpu`, and pays a multi-minute engine build on first run unless engine caching
  is configured. Pass `providers=` to `SonicOnnxPolicy` to opt in.
- **PyTorch** would avoid the CUDA-matching step entirely, since Isaac Lab's torch is already
  correct by construction — this is how `eval_agent_trl.py` runs SONIC during training. It needs the
  `.pt` checkpoint plus the training dependency stack (hydra, trl, transformers), which is a much
  heavier install for a workflow that otherwise needs none of it.
- **Warp** is not applicable. It is a GPU kernel-authoring framework, not an inference runtime: it
  has no ONNX ingestion and no neural-network layers, so the model would have to be reimplemented by
  hand. It ships with Isaac Lab, but that does not make it a fit.

One consequence of the ONNX path: the released graphs are exported with a fixed batch size of 1.

## Known characteristics

**~200 ms reference lag.** SONIC's `smpl` encoder consumes 10 reference frames at 20 ms spacing,
which training supplied as *future* motion. A live operator has no future, so the ten most recent
frames are presented and the policy treats the oldest as "now" — the same approach the C++ deploy
stack takes. This is inherent to the checkpoint; the `low_latency` checkpoint uses a shorter window.

**No operator calibration.** Retargeting runs SMPL forward kinematics on a fixed canonical skeleton
using only the operator's tracked rotations, so operator proportions are replaced by the canonical
body. There is no height or limb-length calibration, matching the shipped PICO implementation.

**No success termination.** `SonicTeleopG1EnvCfg` has no task, so `record_demos.py` cannot mark
demos successful. Subclass it and add a `success` termination term for a real collection task.

**Single environment.** The released SONIC ONNX graphs are exported with a fixed batch size of 1.
Multi-environment rollouts would need a re-export with a dynamic batch axis.

## Why a launcher

Isaac Lab resolves a task by name through `gym.make`, and ids are registered as an import side
effect. `isaaclab_tasks` only auto-imports its own subpackages, and there is no plugin hook for task
packages living outside it, so the stock scripts cannot see this repo's ids on their own.

Isaac Lab's template generator handles this by emitting a project-local runner that imports the
project's tasks and then calls an Isaac Lab library entry point
(`tools/template/templates/external/train`). That shape only covers workflows factored into
`isaaclab_rl.entrypoints` — train, play, zero-agent, random-agent. Teleoperation and demo recording
have no library entry point, so `gear_sonic.lab_teleop.scripts.run` imports the task registry and
then executes the stock script itself. Nothing is vendored or copied, so upstream changes are picked
up automatically.

## Troubleshooting

**The robot is invisible but simulates correctly.** The URDF importer could not resolve
`package://` mesh URIs, so the converted USD has no visual geometry — only collision shapes with
`purpose = "guide"`, which are not rendered. `make_g1_sonic_cfg()` sets `ros_package_paths` to
prevent this; if you build a config by hand, set it too.

**"IsaacTeleop session step failed (XR session likely torn down)".** The teleop session catches
exceptions raised inside retargeter `_compute_fn` and reports them as XR teardowns. The message text
after the colon is the real exception. Common causes are indexing a tensor group with a string
instead of its index enum, and relative asset paths that only resolve from the repo root.

**The robot spins continuously and sinks.** A quaternion convention mismatch. Isaac Lab 3.0 returns
**XYZW**; `gear_sonic`'s rotation helpers are all `w_last=False`, i.e. **WXYZ**. Convert at the
boundary with `gear_sonic.lab_teleop.mdp.actions.isaaclab_quat_to_wxyz`.
