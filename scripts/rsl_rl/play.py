# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

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
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

## --- MODIFICATION START --- ##
# Import matplotlib for plotting
import matplotlib.pyplot as plt
## --- MODIFICATION END --- ##

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import disney_bdx.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
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
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # This section robustly gets the joint names for printing later.
    try:
        robot_articulation = env.unwrapped.scene._articulations["robot"]
        joint_names = robot_articulation.joint_names
    except KeyError:
        print("\n[ERROR] Could not find an articulation named 'robot' in the scene.")
        print("Please check the available names below and replace 'robot' in the script with the correct one.")
        print("Available articulation keys:", list(env.unwrapped.scene._articulations.keys()))
        print("\n")
        simulation_app.close()
        exit()

    num_lines_printed = 0

    ## --- MODIFICATION START --- ##
    # Dictionary to store the history of all tracked data for plotting
    data_history = {
        "timesteps": [],
        "imu_ang_vel": [],
        "projected_gravity": [],
    }
    ## --- MODIFICATION END --- ##

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # compute actions
        with torch.inference_mode():
            actions = policy(obs)
            # step the environment
            obs, _, _, _ = env.step(actions)

        # ANSI escape code to move the cursor up, clearing the previous printout.
        if num_lines_printed > 0:
            print(f"\033[{num_lines_printed}A", end="")

        # Prepare the output string for a clean, in-place display
        output_string = ""
        obs_data = obs[0]

        # --- Extract data from observation tensor ---
        imu_ang_vel = obs_data[0:3]
        projected_gravity = obs_data[3:6]
        velocity_commands = obs_data[6:9]
        joint_pos_obs = obs_data[9:19]
        joint_vel_obs = obs_data[19:29]
        last_actions = obs_data[29:39]
        joint_positions_rad = robot_articulation.data.joint_pos[0]

        ## --- MODIFICATION START --- ##
        # Store current data for plotting later. Move to CPU and convert to numpy.
        data_history["timesteps"].append(timestep)
        data_history["imu_ang_vel"].append(imu_ang_vel.cpu().numpy())
        data_history["projected_gravity"].append(projected_gravity.cpu().numpy())
        ## --- MODIFICATION END --- ##

        # Helper function for clean printing
        def format_array(arr_tensor):
            arr = arr_tensor.cpu().numpy()
            return f"[{' '.join(f'{x:7.3f}' for x in arr)}]"

        # Build the formatted string for printing
        output_string += "--- Observation Tensor Breakdown ---\n"
        output_string += f"IMU Ang Vel       (0:3)  : {format_array(imu_ang_vel)}\n"
        output_string += f"Projected Gravity (3:6)  : {format_array(projected_gravity)}\n"
        output_string += f"Velocity Commands (6:9) : {format_array(velocity_commands)}\n"
        output_string += "------------------------------------\n"
        output_string += "--- Robot State Observations ---\n"
        output_string += f"Joint Pos Obs (9:19): {format_array(joint_pos_obs)}\n"
        output_string += f"Joint Vel Obs (19:29): {format_array(joint_vel_obs)}\n"
        output_string += f"Last Action   (29:39): {format_array(last_actions)}\n"
        output_string += "------------------------------------\n"
        output_string += "--- Joint Positions (Live from Sim) ---\n"
        for i, name in enumerate(joint_names):
            angle_rad = joint_positions_rad[i].item()
            angle_deg = np.rad2deg(angle_rad)
            output_string += f"{name:<20}: {angle_rad:8.4f} rad  ({angle_deg:8.2f} deg)\n"
        output_string += "--------------------------------------"

        # Print the entire block
        print(output_string, flush=True)
        num_lines_printed = output_string.count("\n") + 1

        # for video recording
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
             timestep += 1 # always increment timestep

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)


    ## --- MODIFICATION START --- ##
    # Plotting and saving logic after the simulation loop finishes

    # Define a directory to save the graphs
    graph_save_dir = os.path.join(log_dir, "graphs_play")
    os.makedirs(graph_save_dir, exist_ok=True)
    print(f"\n[INFO] Saving graphs to: {graph_save_dir}")

    timesteps = data_history["timesteps"]
    vector_legends = ["X", "Y", "Z"]

    # --- Graph 1: Projected Gravity vs. Time ---
    plt.figure(figsize=(12, 6))
    projected_gravity_data = np.array(data_history["projected_gravity"])
    for i in range(projected_gravity_data.shape[1]):
        plt.plot(timesteps, projected_gravity_data[:, i], label=f"Component {vector_legends[i]}")
    plt.title("Projected Gravity vs. Time")
    plt.xlabel("Timestep")
    plt.ylabel("Gravity Vector Component")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    projected_gravity_filename = os.path.join(graph_save_dir, "projected_gravity.png")
    plt.savefig(projected_gravity_filename)
    plt.close()
    print(f"[INFO] Saved projected gravity graph to: {projected_gravity_filename}")


    # --- Graph 2: Angular Velocity vs. Time ---
    plt.figure(figsize=(12, 6))
    angular_velocity_data = np.array(data_history["imu_ang_vel"])
    for i in range(angular_velocity_data.shape[1]):
        plt.plot(timesteps, angular_velocity_data[:, i], label=f"Component {vector_legends[i]}")
    plt.title("IMU Angular Velocity vs. Time")
    plt.xlabel("Timestep")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    angular_velocity_filename = os.path.join(graph_save_dir, "angular_velocity.png")
    plt.savefig(angular_velocity_filename)
    plt.close()
    print(f"[INFO] Saved angular velocity graph to: {angular_velocity_filename}")


    print("[INFO] All graphs saved successfully.")
    ## --- MODIFICATION END --- ##

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()