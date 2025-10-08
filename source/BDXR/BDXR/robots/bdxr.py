from pathlib import Path

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg # noqa: F401
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

BDX_CFG = ArticulationCfg(
    # TODO: Change by the URDF directly by adding in the installation setup, the need to curl
    # It will reduce the weight of this repository
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/BDXR/BDX-R.usd",
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
        # TODO: Implement these gains as formula, to understand better where it come from (mention BeyondMimic inspiration)
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

# TODO: Add dynamic scaling