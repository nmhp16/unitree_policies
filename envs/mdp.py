"""Observation, reward, termination, and event functions for the G1 motion-
tracking env. Follows the Isaac Lab manager-based RL function signature:

    fn(env, ...) -> torch.Tensor of shape (num_envs,) or (num_envs, K)

The env is expected to expose:
    env.motion          : MotionReference instance
    env.motion_phase    : (num_envs,) float tensor — current phase per env
    env.scene["robot"]  : Articulation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .g1_tracking_env import G1MotionTrackingEnv


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

def motion_phase(env: "G1MotionTrackingEnv") -> torch.Tensor:
    """(N, 2) — sin/cos of motion phase. Phase encodes 'where in the clip am I'."""
    p = env.motion_phase * 2 * torch.pi
    return torch.stack([p.sin(), p.cos()], dim=-1)


def motion_reference_target(env: "G1MotionTrackingEnv") -> torch.Tensor:
    """(N, J + 7) — target joint pos + target root pos (3) + target root quat (4).

    Feeding the reference into the observation lets the policy condition on
    'what should I look like next' rather than re-deriving it from phase alone.
    Critical for fast convergence on multi-second clips.
    """
    ref = env.query_reference()
    return torch.cat([ref["joint_pos"], ref["root_pos"], ref["root_rot"]], dim=-1)


# -----------------------------------------------------------------------------
# Rewards — DeepMimic-style tracking
# -----------------------------------------------------------------------------

def joint_pos_tracking(
    env: "G1MotionTrackingEnv",
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ref = env.query_reference()
    err = asset.data.joint_pos - ref["joint_pos"]
    return torch.exp(-(err.pow(2).mean(dim=-1)) / (sigma**2))


def joint_vel_tracking(
    env: "G1MotionTrackingEnv",
    sigma: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ref = env.query_reference()
    err = asset.data.joint_vel - ref["joint_vel"]
    return torch.exp(-(err.pow(2).mean(dim=-1)) / (sigma**2))


def root_pos_tracking(
    env: "G1MotionTrackingEnv",
    sigma: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track root height + xy displacement relative to motion start.

    We track root pose *relative to the initial reference frame*, not absolute
    world frame — otherwise an env spawning at world origin can never match a
    reference clip recorded somewhere else.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ref = env.query_reference()
    err = (asset.data.root_pos_w - env.scene.env_origins) - ref["root_pos"]
    return torch.exp(-(err.pow(2).sum(dim=-1)) / (sigma**2))


def root_rot_tracking(
    env: "G1MotionTrackingEnv",
    sigma: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ref = env.query_reference()
    # quat dot product → 0 means orthogonal, 1 means aligned, -1 means flipped
    dot = (asset.data.root_quat_w * ref["root_rot"]).sum(dim=-1).abs()
    angle = 2 * (1 - dot).clamp(min=0).sqrt()  # rough angular distance
    return torch.exp(-(angle.pow(2)) / (sigma**2))


def root_lin_vel_tracking(
    env: "G1MotionTrackingEnv",
    sigma: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ref = env.query_reference()
    err = asset.data.root_lin_vel_w - ref["root_lin_vel"]
    return torch.exp(-(err.pow(2).sum(dim=-1)) / (sigma**2))


def root_ang_vel_tracking(
    env: "G1MotionTrackingEnv",
    sigma: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Crucial for spin / yaw-heavy motions. Without this, root_rot tracking
    alone leaves the *speed* of rotation under-specified — the policy can
    achieve high root_rot reward intermittently by spinning fast and then
    correcting, instead of holding a steady ω."""
    asset: Articulation = env.scene[asset_cfg.name]
    ref = env.query_reference()
    err = asset.data.root_ang_vel_w - ref["root_ang_vel"]
    return torch.exp(-(err.pow(2).sum(dim=-1)) / (sigma**2))


def alive(env: "G1MotionTrackingEnv") -> torch.Tensor:
    """Constant +1 per step. Pairs with early-termination on fall/divergence
    to reward survival explicitly — without it, fragile policies that match
    the reference pose but topple a step later still score well."""
    return torch.ones(env.num_envs, device=env.device)


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

def root_too_low(
    env: "G1MotionTrackingEnv",
    minimum_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def deviated_from_reference(
    env: "G1MotionTrackingEnv",
    joint_threshold: float = 1.2,
    root_threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Early-terminate trajectories that wandered too far from the reference.

    Critical for sample efficiency — without this, the policy spends most of
    its samples on already-doomed rollouts. DeepMimic and AMP both use this.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ref = env.query_reference()
    j_err = (asset.data.joint_pos - ref["joint_pos"]).pow(2).mean(dim=-1).sqrt()
    r_err = ((asset.data.root_pos_w - env.scene.env_origins) - ref["root_pos"]).norm(dim=-1)
    return (j_err > joint_threshold) | (r_err > root_threshold)


def motion_completed(env: "G1MotionTrackingEnv") -> torch.Tensor:
    """For non-loopable clips: terminate when phase reaches 1.0."""
    if env.motion.loopable:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.motion_phase >= 1.0 - 1e-6


# -----------------------------------------------------------------------------
# Events — Reference State Initialization (RSI) + Domain Randomization
# -----------------------------------------------------------------------------

def reset_to_motion_phase(
    env: "G1MotionTrackingEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """RSI: sample a random phase, snap robot pose to the reference at that phase.

    This is the key trick from Peng et al. 2018 (DeepMimic): instead of always
    starting at t=0, sample uniformly across the clip. Every phase gets equal
    coverage even early in training, so the hard middle/late phases aren't
    bottlenecked on first mastering the easy beginning.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)

    # For non-loopable clips, cap the start phase so each episode has a few
    # steps to play out before motion_completed terminates it.
    if env.motion.loopable:
        new_phase = torch.rand(n, device=env.device)
    else:
        new_phase = torch.rand(n, device=env.device) * 0.85
    env.motion_phase[env_ids] = new_phase

    ref = env.motion.index(env.motion_phase[env_ids])
    asset.write_joint_state_to_sim(ref["joint_pos"], ref["joint_vel"], env_ids=env_ids)
    root_pose = torch.cat(
        [ref["root_pos"] + env.scene.env_origins[env_ids], ref["root_rot"]],
        dim=-1,
    )
    root_vel = torch.cat([ref["root_lin_vel"], ref["root_ang_vel"]], dim=-1)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def advance_motion_phase(env: "G1MotionTrackingEnv") -> None:
    """Tick phase forward by one sim-step's worth of clip time."""
    dphase = env.step_dt / env.motion.duration
    env.motion_phase += dphase
    if env.motion.loopable:
        env.motion_phase = env.motion_phase % 1.0
