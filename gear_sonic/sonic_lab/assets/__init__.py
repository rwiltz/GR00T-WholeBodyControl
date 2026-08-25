# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot assets configured for SONIC."""

from gear_sonic.sonic_lab.assets.g1_sonic import (
    G1_ISAACLAB_TO_MUJOCO_MAPPING,
    G1_MODEL_12_ACTION_SCALE,
    G1_SONIC_CFG,
    SONIC_NUM_JOINTS,
    make_g1_sonic_cfg,
)

__all__ = [
    "G1_ISAACLAB_TO_MUJOCO_MAPPING",
    "G1_MODEL_12_ACTION_SCALE",
    "G1_SONIC_CFG",
    "SONIC_NUM_JOINTS",
    "make_g1_sonic_cfg",
]
