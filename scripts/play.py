"""Roll out a trained policy and record an MP4 for visual verification.

    ~/IsaacLab/isaaclab.sh -p scripts/play.py \\
        --task spin_attack --checkpoint outputs/rl_runs/spin_attack/final.pt

Video is recorded by default to outputs/rl_runs/<task>/videos/<timestamp>.mp4.
Pass --no-video to skip recording. Needs ffmpeg on PATH for the mp4 encode.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="track_walk")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--video", action="store_true", default=True,
                    help="Record an MP4 of the rollout (default: on)")
parser.add_argument("--no-video", dest="video", action="store_false")
parser.add_argument("--video_length", type=int, default=400,
                    help="Steps to capture (400 ≈ 8s at 50 Hz control)")
parser.add_argument("--single_episode", action="store_true",
                    help="Stop at first episode termination so the recording "
                         "doesn't include the reset/fall transient between "
                         "episodes. Pads remaining frames with the final state.")
parser.add_argument("--warmup_steps", type=int, default=15,
                    help="Zero-action sim steps before the recording starts. "
                         "Lets physics settle after the RSI hard-write so the "
                         "first recorded frame is a stable pose, not a "
                         "post-reset transient.")
parser.add_argument("--start_phase", type=float, default=None,
                    help="If set, override RSI's random phase sampling and "
                         "force the motion to start at this phase ∈ [0, 1). "
                         "Use 0.0 for a deterministic stand→action→stand demo.")
parser.add_argument("--log_dir", type=str, default="outputs/rl_runs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Off-screen rendering needs cameras; force on when recording.
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from unitree_policies.envs.g1_tracking_env import G1MotionTrackingEnv
from unitree_policies.envs.g1_tracking_env_cfg import G1MotionTrackingEnvCfg
from unitree_policies.tasks import apply_task_yaml
from unitree_policies.configs.ppo_cfg import G1PPORunnerCfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = PROJECT_ROOT / "tasks"


def main() -> None:
    env_cfg = G1MotionTrackingEnvCfg()
    env_cfg = apply_task_yaml(env_cfg, TASK_DIR / f"{args.task}.yaml")
    env_cfg.scene.num_envs = args.num_envs

    env = G1MotionTrackingEnv(cfg=env_cfg, render_mode="rgb_array")

    if args.video:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = Path(args.log_dir) / args.task / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        # Delay video start by `warmup_steps` so the RSI/physics settle frames
        # don't end up in the saved mp4.
        record_start = args.warmup_steps
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == record_start,
            video_length=args.video_length,
            name_prefix=f"play_{stamp}",
            disable_logger=True,
        )
        print(f"[play] recording to {video_dir}/play_{stamp}-step-{record_start}.mp4", flush=True)

    env = RslRlVecEnvWrapper(env)

    agent_cfg = G1PPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    print(f"[play] loaded {args.checkpoint}", flush=True)

    # Optional: snap the motion to a fixed phase before recording so the demo
    # rollout is deterministic (e.g. phase=0 always starts at the motion's
    # first frame). Inlines the snap logic from mdp.reset_to_motion_phase but
    # skips the random-phase sampling so the phase we set actually sticks.
    if args.start_phase is not None:
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.assets import Articulation as _Articulation
        inner = env.unwrapped
        asset_cfg = SceneEntityCfg("robot", joint_names=list(inner.motion.motion.joint_names))
        asset_cfg.resolve(inner.scene)
        asset: _Articulation = inner.scene[asset_cfg.name]
        env_ids = torch.arange(inner.num_envs, device=inner.device)
        inner.motion_phase[:] = float(args.start_phase)
        ref = inner.motion.index(inner.motion_phase[env_ids])
        asset.write_joint_state_to_sim(
            ref["joint_pos"], ref["joint_vel"],
            joint_ids=asset_cfg.joint_ids, env_ids=env_ids,
        )
        root_pose = torch.cat(
            [ref["root_pos"] + inner.scene.env_origins[env_ids], ref["root_rot"]],
            dim=-1,
        )
        root_vel = torch.cat([ref["root_lin_vel"], ref["root_ang_vel"]], dim=-1)
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        print(f"[play] forced start_phase = {args.start_phase}", flush=True)

    # Warmup: step the env with zero actions so RSI write transients settle
    # before recording starts. RecordVideo's step_trigger fires at the matching
    # step, so frames 0..warmup_steps-1 are NOT in the saved mp4.
    obs = env.get_observations()
    if args.warmup_steps > 0:
        zero_action = torch.zeros_like(policy(obs))
        for _ in range(args.warmup_steps):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(zero_action)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
        print(f"[play] warmup done ({args.warmup_steps} zero-action steps)", flush=True)

    steps = 0
    terminated_step: int | None = None
    while sim_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # Reset recurrent / running state for terminated episodes.
            if hasattr(policy, "reset"):
                policy.reset(dones)
        steps += 1
        if steps % 50 == 0:
            print(f"[play] step {steps}/{args.video_length}", flush=True)
        if args.single_episode and terminated_step is None and dones.any().item():
            # Record the moment of termination, then stop forwarding actions.
            # We keep ticking the sim so RecordVideo can pad the remaining
            # frames with the same final state (avoids a hard cut).
            terminated_step = steps
            print(f"[play] episode ended at step {steps}; freezing remainder",
                  flush=True)
        if args.video and steps >= args.video_length:
            break
        if (args.single_episode and terminated_step is not None
                and steps >= terminated_step + 20):
            # Allow a short tail (~0.4s) after termination, then stop early
            # — there's no point recording the fall-and-reset transient.
            break

    # Critical: env.close() finalizes the RecordVideo mp4. Skip it and the
    # video file stays half-written / empty.
    env.close()
    print(f"[play] done ({steps} steps); video flushed to disk.", flush=True)


if __name__ == "__main__":
    main()
    sim_app.close()
