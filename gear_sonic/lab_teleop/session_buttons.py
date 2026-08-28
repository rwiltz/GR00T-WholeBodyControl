# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drive the teleop session from the right controller's face buttons.

    right primary click   (A on a Quest)  start / stop teleoperation
    right secondary click (B on a Quest)  reset the episode

Why this is a wrapper rather than configuration
-----------------------------------------------
Isaac Teleop has a purpose-built mechanism for headset-driven session control -- a second
retargeting graph (``teleop_control_pipeline``) emitting a ``[stopped, paused, running]`` one-hot
and a reset pulse. Isaac Lab does not expose it: :class:`IsaacTeleopCfg` offers only
``control_channel_uuid``, and the pipeline is built privately inside
``TeleopSessionLifecycle.start()``. Supplying our own would mean patching Isaac Lab.

Replacing that pipeline would also be a poor trade. ``request_start``, ``request_stop`` and
``reset`` are all implemented by injecting commands into the message processor that the pipeline
owns, so a replacement returning no processor would silently turn every one of them -- including
the CloudXR client's own controls -- into a no-op.

So this adds a driver alongside it instead, using only public API::

    IsaacTeleopDevice.advance()          wrapped, to run once per step
    lifecycle.last_right_controller      public property, refreshed each step
    device.last_control_events.is_active public property, the current run state
    device.request_start / request_stop / reset(pause=True)

Nothing here changes what ``advance()`` returns, so the
``action = teleop_interface.advance(); env.step(action)`` contract is untouched.
"""

from __future__ import annotations

import logging

from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

__all__ = ["install_session_button_control"]

logger = logging.getLogger(__name__)

#: Set once the wrapper is installed, so repeated env construction does not stack wrappers.
_INSTALLED = False


def _button_down(controller, index: ControllerInputIndex) -> bool:
    """Read one boolean button off a controller tensor group.

    Args:
        controller: ``last_right_controller`` tensor group. Never ``None`` here -- the caller
            skips the whole poll when tracking is absent.
        index: Field to read.

    Returns:
        ``True`` when the button is pressed. An unreadable field reads as not pressed.
    """
    try:
        return bool(controller[index])
    except Exception:  # noqa: BLE001 - a malformed group must not take down the session
        logger.debug("could not read controller field %s", index, exc_info=True)
        return False


class _ButtonState:
    """Rising-edge latches for the two buttons.

    Edge-triggered rather than level-triggered: holding A would otherwise start and stop the
    session on every control step for as long as the thumb rests on it.
    """

    def __init__(self) -> None:
        self.primary_was_down = False
        self.secondary_was_down = False


def _poll(device, state: _ButtonState) -> None:
    """Apply one step's worth of button edges to the session."""
    controller = device._session_lifecycle.last_right_controller  # noqa: SLF001 - see module note
    if controller is None:
        # Tracking is absent, which says nothing about the buttons. Treating it as "released"
        # would let a momentary dropout while the operator holds A read as a release followed by
        # a fresh press, toggling the session off behind their back. Hold the latches instead:
        # a real release is only believed when a tracked controller reports one.
        return

    primary = _button_down(controller, ControllerInputIndex.PRIMARY_CLICK)
    if primary and not state.primary_was_down:
        # Toggle against the session's own state rather than a local flag, so the CloudXR client
        # and this button cannot drift apart.
        is_active = device.last_control_events.is_active
        if is_active:
            device.request_stop()
            print("[SONIC] teleop stopped (right A)", flush=True)
        else:
            device.request_start()
            print("[SONIC] teleop started (right A)", flush=True)
    state.primary_was_down = primary

    secondary = _button_down(controller, ControllerInputIndex.SECONDARY_CLICK)
    if secondary and not state.secondary_was_down:
        # pause=True is the operator reset: come back paused rather than running straight into the
        # next episode, matching what keyboard 'R' does.
        device.reset(pause=True)
        print("[SONIC] reset (right B)", flush=True)
    state.secondary_was_down = secondary


def install_session_button_control() -> None:
    """Wrap :meth:`IsaacTeleopDevice.advance` so the right controller drives the session.

    Idempotent: environments are constructed more than once in a process (replay batches, tests),
    and stacking wrappers would poll the buttons several times per step and swallow every second
    edge.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from isaaclab_teleop import IsaacTeleopDevice

    original_advance = IsaacTeleopDevice.advance

    def advance_with_buttons(self, *args, **kwargs):
        result = original_advance(self, *args, **kwargs)
        state = getattr(self, "_sonic_button_state", None)
        if state is None:
            state = _ButtonState()
            self._sonic_button_state = state
        try:
            _poll(self, state)
        except Exception:  # noqa: BLE001 - button handling must never break the control loop
            logger.warning("session button polling failed; continuing", exc_info=True)
        return result

    IsaacTeleopDevice.advance = advance_with_buttons
    _INSTALLED = True
