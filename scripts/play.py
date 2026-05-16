"""Roll out a trained policy and record an MP4 for visual verification.

    ~/IsaacLab/isaaclab.sh -p scripts/play.py \\
        --task spin_attack --checkpoint outputs/rl_runs/spin_attack/final.pt

Video is recorded by default to outputs/rl_runs/<task>/videos/<timestamp>.mp4.
Pass --no-video to skip recording (faster, GUI-only playback). The video
captures env 0's viewport — increase --num_envs to also see neighbors tile
into the frame.

Requires ffmpeg on PATH for the mp4 encode (Isaac Lab's bundled python comes
with imageio[ffmpeg]; if encoding fails install with `apt install ffmpeg`).
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="track_walk")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4,
                    help="More envs make a nicer video tile but use more memory")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--video", action="store_true", default=True,
                    help="Record an MP4 of the rollout (default: on)")
parser.add_argument("--no-video", dest="video", action="store_false",
                    help="Skip video recording (faster, GUI-only)")
parser.add_argument("--video_length", type=int, default=500,
                    help="Steps to capture (500 ≈ 10s at 50 Hz control)")
parser.add_argument("--log_dir", type=str, default="outputs/rl_runs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Off-screen rendering requires cameras to be enabled. Force it on when
# recording video, otherwise the GUI viewport is the only render source and
# headless runs would silently produce no frames.
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from unitree_policies.envs.g1_tracking_env import G1MotionTrackingEnv
from unitree_policies.envs.g1_tracking_env_cfg import G1MotionTrackingEnvCfg
from unitree_policies.tasks import apply_task_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = PROJECT_ROOT / "tasks"
PPO_CFG_PATH = PROJECT_ROOT / "configs" / "ppo_cfg.yaml"


def main() -> None:
    env_cfg = G1MotionTrackingEnvCfg()
    env_cfg = apply_task_yaml(env_cfg, TASK_DIR / f"{args.task}.yaml")
    env_cfg.scene.num_envs = args.num_envs

    env = G1MotionTrackingEnv(cfg=env_cfg, render_mode="rgb_array")

    video_path: Path | None = None
    if args.video:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = Path(args.log_dir) / args.task / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"play_{stamp}"
        # RecordVideo writes <prefix>-episode-N.mp4 files; the wrapper handles
        # the actual encoding via imageio + ffmpeg.
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            name_prefix=f"play_{stamp}",
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env)

    with open(PPO_CFG_PATH) as f:
        ppo_cfg = yaml.safe_load(f)
    runner = OnPolicyRunner(env, ppo_cfg, log_dir=None, device=ppo_cfg["device"])
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=ppo_cfg["device"])

    obs, _ = env.get_observations()
    eps_done = 0
    steps = 0
    while eps_done < args.num_episodes and sim_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        eps_done += int(dones.sum().item())
        steps += 1
        # Stop once we've captured the requested video length even if more
        # episodes are still in flight — RecordVideo would otherwise keep
        # buffering frames it never writes.
        if args.video and steps >= args.video_length:
            break

    if video_path is not None:
        print(f"\nVideo saved under: {video_path.parent}")
        print(f"  Look for: play_{video_path.name.split('_', 1)[1]}-episode-*.mp4")


if __name__ == "__main__":
    main()
    sim_app.close()
