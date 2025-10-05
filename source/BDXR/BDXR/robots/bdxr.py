# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Disney Research robots.

The following configuration parameters are available:

* :obj:`BDX_CFG`: The BD-X robot with implicit Actuator model

Reference:

* https://github.com/rimim/AWD/tree/main/awd/data/assets/go_bdx

"""
from pathlib import Path

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

BDX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/BDXR/BDX-R.usd",
        # usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/Disney/BDX/BDXREACH.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30846),
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip_Yaw", ".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee", ".*_Ankle"],
            stiffness={
                ".*_Hip_Yaw": 78.0,
                ".*_Hip_Roll": 78.0,
                ".*_Hip_Pitch": 78.0,
                ".*_Knee": 78.0,
                ".*_Ankle": 17.0,
            },
            damping={
                ".*_Hip_Yaw": 5.0,
                ".*_Hip_Roll": 5.0,
                ".*_Hip_Pitch": 5.0,
                ".*_Knee": 5.0,
                ".*_Ankle": 1.0,
            },
            armature={
                ".*_Hip_Yaw": 0.02,
                ".*_Hip_Roll": 0.02,
                ".*_Hip_Pitch": 0.02,
                ".*_Knee": 0.02,
                ".*_Ankle": 0.0042,
            },
            effort_limit_sim={
                ".*_Hip_Yaw": 42.0,
                ".*_Hip_Roll": 42.0,
                ".*_Hip_Pitch": 42.0,
                ".*_Knee": 42.0,
                ".*_Ankle": 11.9,
            },
            velocity_limit_sim={
                ".*_Hip_Yaw": 18.849,
                ".*_Hip_Roll": 18.849,
                ".*_Hip_Pitch": 18.849,
                ".*_Knee": 18.849,
                ".*_Ankle": 37.699,
            },
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Disney BD-X robot with implicit actuator model."""
