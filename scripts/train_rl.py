"""Train a motion-tracking policy with rsl_rl PPO.

Run via Isaac Lab's bundled python (which carries the sim bindings):

    ~/IsaacLab/isaaclab.sh -p scripts/train_rl.py \\
        --task track_walk --num_envs 4096 --headless

The task name maps to tasks/<task>.yaml. YAML overrides are overlaid onto the
default G1MotionTrackingEnvCfg before env construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Isaac Sim app launch must happen before importing any isaaclab module.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="track_walk",
                    help="YAML stem under tasks/ (e.g. track_walk)")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override num_envs from the YAML")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override max PPO iterations")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from a saved policy")
parser.add_argument("--log_dir", type=str, default="outputs/rl_runs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import importlib.metadata as metadata

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from unitree_policies.envs.g1_tracking_env import G1MotionTrackingEnv
from unitree_policies.envs.g1_tracking_env_cfg import G1MotionTrackingEnvCfg
from unitree_policies.tasks import apply_task_yaml
from unitree_policies.configs.ppo_cfg import G1PPORunnerCfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = PROJECT_ROOT / "tasks"


def main() -> None:
    task_yaml = TASK_DIR / f"{args.task}.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"No task YAML at {task_yaml}")

    env_cfg = G1MotionTrackingEnvCfg()
    env_cfg = apply_task_yaml(env_cfg, task_yaml)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = G1MotionTrackingEnv(
        cfg=env_cfg,
        render_mode="rgb_array" if not args.headless else None,
    )
    env = RslRlVecEnvWrapper(env)

    agent_cfg = G1PPORunnerCfg()
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    # Migrate cfg fields across rsl-rl version boundaries (drops deprecated
    # MLP fields like `stochastic` that newer rsl-rl no longer accepts).
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    log_dir = Path(args.log_dir) / args.task
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(),
        log_dir=str(log_dir), device=agent_cfg.device,
    )
    if args.checkpoint:
        runner.load(args.checkpoint)
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    runner.save(str(log_dir / "final.pt"))
    print(f"\nSaved final policy to {log_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
    sim_app.close()
