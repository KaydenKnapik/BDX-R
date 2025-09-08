# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from typing import Dict, Optional, Tuple

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .bdxr_velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils.math import quat_from_euler_xyz
import torch
from isaaclab.envs import ManagerBasedEnv
# from .bdxr_rewards import bipedal_air_time_reward, foot_clearance_reward, foot_slip_penalty, joint_position_penalty
from isaaclab.managers import EventTermCfg as EventTerm
import BDXR.tasks.velocity.mdp as mdp
# from . import mdp

##
# Pre-defined configs
##

from BDXR.robots.bdxr import BDX_CFG  # isort:skip

##
# Scene definition
##

##
# MDP settings
##

def print_robot_joint_info(env, entity_cfg: SceneEntityCfg):
    """
    An event function to print the robot's joint order and default positions
    once at the very beginning of the simulation.
    """
    # This is a simple flag to ensure this function's body only runs one time.
    if not hasattr(env, '_joint_info_printed'):
        robot = env.scene[entity_cfg.name]
        
        joint_names_in_order = robot.data.joint_names
        default_joint_pos = robot.data.default_joint_pos[0] # Get for the first env

        print("\n" + "="*40)
        print("      ROBOT JOINT CONFIGURATION (GROUND TRUTH)")
        print("="*40)
        print("This is the exact joint order and default positions for the policy.")
        
        if joint_names_in_order:
            for i, name in enumerate(joint_names_in_order):
                default_pos_value = default_joint_pos[i].item()
                print(f"  Index {i:<2} | Joint Name: {name:<20} | Default Pos: {default_pos_value:.4f}")
        else:
            print("Could not retrieve joint names from the live environment.")
            
        print("="*40 + "\n")
        
        # Set the flag so this block never runs again.
        env._joint_info_printed = True

@configclass
class BDXRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    air_time = RewTerm(
        func=mdp.bipedal_air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
        },
    )
    # penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_Ankle")},
    )
    # penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=2,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot"),
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
            "threshold": 1.0,
        },
    )
    base_height_deviation = RewTerm(
        func=mdp.base_height_l2,
        weight=-2,  # Tune this weight as needed
        params={
            "target_height": 0.30846,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["base_link"]),
        },
    )
    joint_pos = RewTerm(
        func=joint_position_penalty,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )


##
# Environment configuration
##


@configclass
class BDXRFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """BDXR flat environment configuration."""

    rewards: BDXRewards = BDXRewards()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = BDX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/IMU_Mount",  # change if needed
        debug_vis=True)

        # actions
        self.actions.joint_pos.scale = 0.5

        # events
        self.events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        #self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.physics_material.params["static_friction_range"] = (0.1, 2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 2)
        self.events.physics_material.params["asset_cfg"].body_names = ".*_Foot"
        self.events.randomize_imu_mount = EventTerm(
            func=mdp.randomize_imu_mount,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
                "pos_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                },
                "rot_range": {
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
            },
        )
        

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]

        # rewards
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        self.rewards.action_rate_l2.weight =-0.05     # A significant penalty on action rate
        self.rewards.dof_acc_l2.weight =-1.25e-7      # A significant penalty on acceleration
        self.rewards.flat_orientation_l2.weight = -2

        # Walk
        self.commands.base_velocity.ranges.lin_vel_x = (0,0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.7, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)

        # change terrain
        #self.scene.terrain.max_init_terrain_level = None
        ## reduce the number of terrains to save memory
        #if self.scene.terrain.terrain_generator is not None:
        #    self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.01)
        #    self.scene.terrain.terrain_generator.num_rows = 5
        #    self.scene.terrain.terrain_generator.num_cols = 5
        #    self.scene.terrain.terrain_generator.curriculum = False


        # change terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


        