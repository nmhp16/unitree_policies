"""Register gym IDs from YAML task files in this directory.

Drop a `<name>.yaml` next to this file and it becomes available as
`Isaac-G1-Tracking-<Name>-v0` to gym.make() and rsl_rl. No code edits needed.

The YAML is loaded twice:
  - at registration time, only to learn the gym_id
  - at env construction time, to apply overrides to the env_cfg (see
    scripts/train_rl.py, which calls apply_task_yaml())
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import yaml

from ..envs.g1_tracking_env import G1MotionTrackingEnv
from ..envs.g1_tracking_env_cfg import G1MotionTrackingEnvCfg

_HERE = Path(__file__).parent


def _register_all():
    for yaml_path in sorted(_HERE.glob("*.yaml")):
        with open(yaml_path) as f:
            spec = yaml.safe_load(f) or {}
        gym_id = spec.get("gym_id")
        if not gym_id:
            gym_id = f"Isaac-G1-Tracking-{yaml_path.stem.replace('_', '-').title()}-v0"
        if gym_id in gym.registry:
            continue
        gym.register(
            id=gym_id,
            entry_point="unitree_policies.envs.g1_tracking_env:G1MotionTrackingEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "unitree_policies.envs.g1_tracking_env_cfg:G1MotionTrackingEnvCfg"
                ),
                "task_yaml": str(yaml_path),
            },
        )


def apply_task_yaml(cfg: G1MotionTrackingEnvCfg, yaml_path: str | Path) -> G1MotionTrackingEnvCfg:
    """Overlay the task YAML onto a default env_cfg. Called from train_rl.py."""
    with open(yaml_path) as f:
        spec = yaml.safe_load(f) or {}

    # Motion
    motion = spec.get("motion", {})
    cfg.motion_path = motion.get("path", cfg.motion_path)
    cfg.motion_fps_override = motion.get("fps_override")

    # Episode
    ep = spec.get("episode", {})
    cfg.episode_length_s = ep.get("length_s", cfg.episode_length_s)

    # Reward weights — overlay onto whatever the default cfg has
    for term_name, weight in (spec.get("rewards", {}) or {}).items():
        if hasattr(cfg.rewards, term_name):
            getattr(cfg.rewards, term_name).weight = float(weight)

    # Termination thresholds
    term = spec.get("terminations", {}) or {}
    if "minimum_height" in term and hasattr(cfg.terminations, "fell"):
        cfg.terminations.fell.params["minimum_height"] = float(term["minimum_height"])
    if "joint_threshold" in term and hasattr(cfg.terminations, "diverged"):
        cfg.terminations.diverged.params["joint_threshold"] = float(term["joint_threshold"])
    if "root_threshold" in term and hasattr(cfg.terminations, "diverged"):
        cfg.terminations.diverged.params["root_threshold"] = float(term["root_threshold"])

    # Scene size
    scene = spec.get("scene", {}) or {}
    if "num_envs" in scene:
        cfg.scene.num_envs = int(scene["num_envs"])
    if "env_spacing" in scene:
        cfg.scene.env_spacing = float(scene["env_spacing"])

    return cfg


_register_all()
