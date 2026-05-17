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
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            name_prefix=f"play_{stamp}",
            disable_logger=True,
        )
        print(f"[play] recording to {video_dir}/play_{stamp}-episode-0.mp4", flush=True)

    env = RslRlVecEnvWrapper(env)

    agent_cfg = G1PPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    print(f"[play] loaded {args.checkpoint}", flush=True)

    obs = env.get_observations()
    steps = 0
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
        if args.video and steps >= args.video_length:
            break

    # Critical: env.close() finalizes the RecordVideo mp4. Skip it and the
    # video file stays half-written / empty.
    env.close()
    print(f"[play] done ({steps} steps); video flushed to disk.", flush=True)


if __name__ == "__main__":
    main()
    sim_app.close()
