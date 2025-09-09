# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import BDXR.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# --- MODIFICATION START --- #
# Import matplotlib for plotting
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# from isaaclab_rl.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlVecEnvWrapper,
#     export_policy_as_jit,
#     export_policy_as_onnx,
# )
# from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
# from rsl_rl.runners import OnPolicyRunner

# --- MODIFICATION END --- #


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

    # --- MODIFICATION START --- #
    # Dictionary to store the history of all tracked data for plotting
    # data_history = {
    #     "timesteps": [],
    #     "imu_ang_vel": [],
    #     "projected_gravity": [],
    # }
    # --- MODIFICATION END --- #

    # ANSI escape code to move the cursor up, clearing the previous printout.
    # if num_lines_printed > 0:
    #     print(f"\033[{num_lines_printed}A", end="")

    # Prepare the output string for a clean, in-place display
    # output_string = ""
    # obs_data = obs[0]

    # --- Extract data from observation tensor ---
    # imu_ang_vel = obs_data[0:3]
    # projected_gravity = obs_data[3:6]
    # velocity_commands = obs_data[6:9]
    # joint_pos_obs = obs_data[9:19]
    # joint_vel_obs = obs_data[19:29]
    # last_actions = obs_data[29:39]
    # joint_positions_rad = robot_articulation.data.joint_pos[0]

    # --- MODIFICATION START --- #
    # Store current data for plotting later. Move to CPU and convert to numpy.
    # data_history["timesteps"].append(timestep)
    # data_history["imu_ang_vel"].append(imu_ang_vel.cpu().numpy())
    # data_history["projected_gravity"].append(projected_gravity.cpu().numpy())
    # --- MODIFICATION END --- #

    # Helper function for clean printing
    # def format_array(arr_tensor):
    #     arr = arr_tensor.cpu().numpy()
    #     return f"[{' '.join(f'{x:7.3f}' for x in arr)}]"

    # Build the formatted string for printing
    # output_string += "--- Observation Tensor Breakdown ---\n"
    # output_string += f"IMU Ang Vel       (0:3)  : {format_array(imu_ang_vel)}\n"
    # output_string += f"Projected Gravity (3:6)  : {format_array(projected_gravity)}\n"
    # output_string += f"Velocity Commands (6:9) : {format_array(velocity_commands)}\n"
    # output_string += "------------------------------------\n"
    # output_string += "--- Robot State Observations ---\n"
    # output_string += f"Joint Pos Obs (9:19): {format_array(joint_pos_obs)}\n"
    # output_string += f"Joint Vel Obs (19:29): {format_array(joint_vel_obs)}\n"
    # output_string += f"Last Action   (29:39): {format_array(last_actions)}\n"
    # output_string += "------------------------------------\n"
    # output_string += "--- Joint Positions (Live from Sim) ---\n"
    # for i, name in enumerate(joint_names):
    #     angle_rad = joint_positions_rad[i].item()
    #     angle_deg = np.rad2deg(angle_rad)
    #     output_string += f"{name:<20}: {angle_rad:8.4f} rad  ({angle_deg:8.2f} deg)\n"
    # output_string += "--------------------------------------"

    # Print the entire block
    # print(output_string, flush=True)
    # num_lines_printed = output_string.count("\n") + 1

    # --- MODIFICATION START --- #
    # Plotting and saving logic after the simulation loop finishes

    # Define a directory to save the graphs
    # graph_save_dir = os.path.join(log_dir, "graphs_play")
    # os.makedirs(graph_save_dir, exist_ok=True)
    # print(f"\n[INFO] Saving graphs to: {graph_save_dir}")

    # timesteps = data_history["timesteps"]
    # vector_legends = ["X", "Y", "Z"]

    # # --- Graph 1: Projected Gravity vs. Time ---
    # plt.figure(figsize=(12, 6))
    # projected_gravity_data = np.array(data_history["projected_gravity"])
    # for i in range(projected_gravity_data.shape[1]):
    #     plt.plot(timesteps, projected_gravity_data[:, i], label=f"Component {vector_legends[i]}")
    # plt.title("Projected Gravity vs. Time")
    # plt.xlabel("Timestep")
    # plt.ylabel("Gravity Vector Component")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # projected_gravity_filename = os.path.join(graph_save_dir, "projected_gravity.png")
    # plt.savefig(projected_gravity_filename)
    # plt.close()
    # print(f"[INFO] Saved projected gravity graph to: {projected_gravity_filename}")

    # # --- Graph 2: Angular Velocity vs. Time ---
    # plt.figure(figsize=(12, 6))
    # angular_velocity_data = np.array(data_history["imu_ang_vel"])
    # for i in range(angular_velocity_data.shape[1]):
    #     plt.plot(timesteps, angular_velocity_data[:, i], label=f"Component {vector_legends[i]}")
    # plt.title("IMU Angular Velocity vs. Time")
    # plt.xlabel("Timestep")
    # plt.ylabel("Angular Velocity (rad/s)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # angular_velocity_filename = os.path.join(graph_save_dir, "angular_velocity.png")
    # plt.savefig(angular_velocity_filename)
    # plt.close()
    # print(f"[INFO] Saved angular velocity graph to: {angular_velocity_filename}")

    # print("[INFO] All graphs saved successfully.")
    # --- MODIFICATION END --- #
