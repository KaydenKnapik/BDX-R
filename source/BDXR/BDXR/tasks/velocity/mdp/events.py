
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_imu_mount(
    env: ManagerBasedEnv,
    env_ids: Optional[torch.Tensor],
    sensor_cfg: SceneEntityCfg,
    pos_range: Dict[str, Tuple[float, float]],
    rot_range: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Helper to randomise the IMU's local pose on every env reset."""
    imu_sensor = env.scene.sensors[sensor_cfg.name]

    # Get the envs which reset
    env_indices = (
        env_ids
        if env_ids is not None
        else torch.arange(imu_sensor.num_instances, device=env.device)
    )
    num_envs_to_update = len(env_indices)

    def sample_uniform(lo: float, hi: float) -> torch.Tensor:
        """Return `num_envs_to_update` samples from [lo, hi)."""
        return (hi - lo) * torch.rand(num_envs_to_update, device=env.device) + lo

    # Sample translation offsets
    position_offsets: torch.Tensor = torch.stack(
        [
            sample_uniform(*pos_range["x"]),
            sample_uniform(*pos_range["y"]),
            sample_uniform(*pos_range["z"]),
        ],
        dim=-1,  # shape = (N, 3)
    )

    # Sample orientation offsets
    roll_offsets = sample_uniform(*rot_range["roll"])
    pitch_offsets = sample_uniform(*rot_range["pitch"])
    yaw_offsets = sample_uniform(*rot_range["yaw"])

    quaternion_offsets: torch.Tensor = quat_from_euler_xyz(
        roll_offsets, pitch_offsets, yaw_offsets  # shape = (N, 4)
    )

    # Write the offsets into the sensor’s internal buffers
    imu_sensor._offset_pos_b[env_indices] = position_offsets
    imu_sensor._offset_quat_b[env_indices] = quaternion_offsets

    # Return summary scalars for logging / curriculum
    # Not sure if this is needed
    mean_offset_cm: float = (position_offsets.norm(dim=-1).mean() * 100.0).item()
    mean_tilt_deg: float = (
        torch.rad2deg(torch.acos(quaternion_offsets[:, 0].clamp(-1.0, 1.0)))
        .mean()
        .item()
    )

    return {
        "imu_offset_cm": mean_offset_cm,
        "imu_tilt_deg": mean_tilt_deg,
    }