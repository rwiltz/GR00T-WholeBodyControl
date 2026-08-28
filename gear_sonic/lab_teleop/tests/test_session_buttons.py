# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Right-controller session control: A starts/stops, B resets.

The install step is not covered here because it imports ``isaaclab_teleop``, which needs Isaac
Sim's ``carb``. The edge logic is where the bugs live and it is pure Python, so it is exercised
against a stand-in device.
"""

from __future__ import annotations

from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

from gear_sonic.lab_teleop.session_buttons import _ButtonState, _poll


class _Controller(dict):
    """Tensor-group stand-in: unpressed buttons read as 0."""

    def __getitem__(self, key):  # noqa: ANN001, ANN204
        return self.get(int(key), 0)


class _Events:
    is_active = False


class _Device:
    """Records the session calls a real device would make."""

    last_right_controller = None

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_control_events = _Events()
        self._session_lifecycle = self

    def request_start(self) -> None:
        self.calls.append("start")
        self.last_control_events.is_active = True

    def request_stop(self) -> None:
        self.calls.append("stop")
        self.last_control_events.is_active = False

    def reset(self, pause: bool = False) -> None:
        self.calls.append(f"reset(pause={pause})")


def _run(frames: list[dict | None]) -> list[str]:
    device, state = _Device(), _ButtonState()
    for frame in frames:
        device.last_right_controller = None if frame is None else _Controller(frame)
        _poll(device, state)
    return device.calls


def test_primary_click_toggles_the_session_once_per_press() -> None:
    """Holding A must not free-run the toggle at the control rate."""
    down = {int(ControllerInputIndex.PRIMARY_CLICK): 1}
    assert _run([{}, down, down, down, {}, down, {}]) == ["start", "stop"]


def test_toggle_follows_the_session_state_not_a_local_flag() -> None:
    """A press while already running must stop, even though this driver never started it.

    The CloudXR client can start and stop the session too. Toggling a local boolean would drift
    out of step with it and invert the button.
    """
    device, state = _Device(), _ButtonState()
    device.last_control_events.is_active = True  # started elsewhere
    device.last_right_controller = _Controller({int(ControllerInputIndex.PRIMARY_CLICK): 1})
    _poll(device, state)
    assert device.calls == ["stop"]


def test_secondary_click_resets_paused() -> None:
    """B resets and comes back paused, matching the operator reset keyboard 'R' performs."""
    down = {int(ControllerInputIndex.SECONDARY_CLICK): 1}
    assert _run([{}, down, down, {}]) == ["reset(pause=True)"]


def test_a_tracking_dropout_does_not_fake_a_release() -> None:
    """Losing the controller must not toggle the session behind the operator's back.

    If absence counted as "released", a momentary dropout while A is held would read as a release
    followed by a fresh press on reacquisition -- stopping a running session with no input from
    the operator. The latch is held through the gap instead.
    """
    down = {int(ControllerInputIndex.PRIMARY_CLICK): 1}
    # Pressed, tracking lost while still held, reacquired still held: one action, not two.
    assert _run([down, None, down]) == ["start"]
    # Never pressed, tracking flapping: nothing at all.
    assert _run([{}, None, {}, None]) == []
    # A genuine release while tracked is still believed.
    assert _run([down, None, down, {}, down]) == ["start", "stop"]


def test_the_two_buttons_are_independent() -> None:
    """Pressing both together does both, and neither latches the other."""
    both = {
        int(ControllerInputIndex.PRIMARY_CLICK): 1,
        int(ControllerInputIndex.SECONDARY_CLICK): 1,
    }
    assert _run([{}, both, {}]) == ["start", "reset(pause=True)"]
