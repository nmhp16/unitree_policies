"""Snap G1 to the task's reference pose and save a single render to PNG.

Used to verify the reference motion looks like what you expect (e.g. T-pose
for spin_attack), independent of any trained policy.

    ~/IsaacLab/isaaclab.sh -p scripts/inspect_reference.py \\
        --task spin_attack --enable_cameras --headless
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="spin_attack")
parser.add_argument("--phase", type=float, default=0.0,
                    help="motion phase in [0, 1) to render")
parser.add_argument("--output", default="outputs/reference_frame.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch
import imageio.v2 as imageio

from isaaclab.managers import SceneEntityCfg
from unitree_policies.envs.g1_tracking_env import G1MotionTrackingEnv
from unitree_policies.envs.g1_tracking_env_cfg import G1MotionTrackingEnvCfg
from unitree_policies.envs import mdp
from unitree_policies.tasks import apply_task_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = PROJECT_ROOT / "tasks"


def main():
    env_cfg = G1MotionTrackingEnvCfg()
    env_cfg = apply_task_yaml(env_cfg, TASK_DIR / f"{args.task}.yaml")
    env_cfg.scene.num_envs = 1
    env = G1MotionTrackingEnv(cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    # Force phase to the requested value, then re-snap the robot pose to match.
    env.motion_phase[:] = args.phase
    asset_cfg = SceneEntityCfg("robot", joint_names=list(env.motion.motion.joint_names))
    asset_cfg.resolve(env.scene)
    mdp.reset_to_motion_phase(env, torch.tensor([0], device=env.device), asset_cfg)

    # Take several zero-action steps so the renderer warms up and captures
    # the post-write state. The first few render frames after AppLauncher are
    # commonly blank — keep stepping until we get a non-black frame or hit a
    # step cap.
    actions = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    frame = None
    for _ in range(20):
        env.step(actions)
        # Re-snap each step so physics integration doesn't drift the joints away
        # from the reference pose we want to inspect.
        env.motion_phase[:] = args.phase
        mdp.reset_to_motion_phase(env, torch.tensor([0], device=env.device), asset_cfg)
        frame = env.render()
        if frame is not None and frame.sum() > 0:
            break

    if frame is None or frame.sum() == 0:
        raise RuntimeError(
            "Render returned blank — check --enable_cameras and that a camera "
            "is configured in the env_cfg.scene."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, frame)
    print(f"[inspect] saved {out_path} (shape={frame.shape})")


if __name__ == "__main__":
    main()
    sim_app.close()
