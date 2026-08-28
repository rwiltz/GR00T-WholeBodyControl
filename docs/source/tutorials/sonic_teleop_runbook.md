# SONIC teleop runbook

Get a second machine running. For the controls, see
[the control card](sonic_teleop_controls.md).

## 1. Install

```bash
git clone https://github.com/isaac-sim/IsaacLab.git      # 3.0.0-beta2 or newer, source checkout
cd IsaacLab && uv run isaaclab --help                    # creates .venv
source $(pwd)/.venv/bin/activate

cd /path/to/GR00T-WholeBodyControl
pip install --no-deps -e ./gear_sonic                    # --no-deps is deliberate
uv pip install --python $(which python) onnxruntime-gpu==1.22.0
```

The `onnxruntime-gpu` version must match your torch CUDA major: `1.22` for cu12x, `1.27+` for
cu13. A plain `onnxruntime` anywhere in the environment shadows it — same module name, last one
wins — and the session now refuses to start rather than running 24x slower in silence.

## 2. Fetch checkpoints

```bash
python download_from_hf.py --sonic-v1-1     # v1.1
python download_from_hf.py --low-latency    # low-latency + the motion planner
```

The planner ships with `--low-latency` and is needed for walking mode in **every** env.

## 3. Check

```bash
python check_environment.py --lab-teleop
```

Must report `onnxruntime CUDA provider`. Everything else is downstream of that.

## 4. Run

```bash
python -m gear_sonic.lab_teleop.scripts.run teleop \
    --task IsaacContrib-Teleop-Sonic-WholeBody-G1-LowLatency-v0 \
    --viz kit --xr --device cpu
```

Connect the headset when the console prints `CloudXR runtime auto-launched`. `--device cpu` puts
*physics* on the CPU; the policy still runs on the GPU.

Replay a capture instead of wearing a headset:

```bash
python -m gear_sonic.lab_teleop.scripts.run replay \
    --task IsaacContrib-Teleop-Sonic-WholeBody-G1-LowLatency-Bare-Replay-v0 \
    --replay_file ./captures/session.mcap --viz kit --device cpu
```

## Environments

| task id | checkpoint | input | props |
|---|---|---|---|
| `...-G1-v0` | v1.1 | headset | yes |
| `...-G1-Replay-v0` | v1.1 | MCAP | yes |
| `...-G1-LowLatency-v0` | low-latency | headset | yes |
| `...-G1-LowLatency-Replay-v0` | low-latency | MCAP | yes |
| `...-G1-LowLatency-Bare-v0` | low-latency | headset | no |
| `...-G1-LowLatency-Bare-Replay-v0` | low-latency | MCAP | no |

Prefix every id with `IsaacContrib-Teleop-Sonic-WholeBody`.

Live and replay are **not** interchangeable: replay needs a vendor-less tracker pipeline and fails
outright against a live task. The low-latency checkpoint trails you by ~80 ms against v1.1's
~200 ms. `Bare` drops the packing table and its crates.

## When it goes wrong

**Refuses to start, naming onnxruntime.** The GPU wheel got shadowed. The error prints the exact
uninstall/install pair.

**Robot stands in its default pose and ignores you.** No body tracking. The reference arrives
invalid and the controller holds the default pose rather than driving from a collapsed skeleton.
Body tracking needs a PICO; a Quest does not provide it.

**Console prints `[SONIC] mode ...` but nothing moves.** Mode switching is reaching the pipeline,
so the problem is downstream — checkpoint or planner, not input.

**Want to see the mode transition in detail.** Set `debug_transitions=True` on the action config:
it prints the handoff discontinuity, planned velocities and every clock in play.
