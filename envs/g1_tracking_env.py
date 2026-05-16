"""ManagerBasedRLEnv subclass that owns the motion reference + phase state.

The base ManagerBasedRLEnv has no concept of "what reference clip am I
tracking, and where am I in it". We add:

  - `self.motion`: MotionReference (loaded from YAML-configured path)
  - `self.motion_phase`: (num_envs,) phase in [0, 1)
  - `query_reference()`: cached per-step query result so observation +
    reward functions all see the same reference state without recomputing
  - phase advances at the end of each step
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .motion_reference import MotionReference
from . import mdp


class G1MotionTrackingEnv(ManagerBasedRLEnv):
    cfg: "G1MotionTrackingEnvCfg"

    def __init__(self, cfg, **kwargs):
        # Load motion reference before super().__init__ so observation and
        # reward managers can resolve env.motion / env.motion_phase at setup.
        self.motion = _load_motion(
            cfg.motion_path, cfg.motion_fps_override, device=cfg.sim.device
        )
        super().__init__(cfg=cfg, **kwargs)
        self.motion_phase = torch.zeros(self.num_envs, device=self.device)
        self._ref_cache: dict[str, torch.Tensor] | None = None
        self._ref_cache_step = -1

    @property
    def step_dt(self) -> float:
        return float(self.cfg.sim.dt * self.cfg.decimation)

    def query_reference(self) -> dict[str, torch.Tensor]:
        """Per-step cached reference query — all manager terms see the same state."""
        if self._ref_cache_step != self.common_step_counter:
            self._ref_cache = self.motion.query(self.motion_phase)
            self._ref_cache_step = self.common_step_counter
        return self._ref_cache

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        mdp.advance_motion_phase(self)
        return obs, rew, term, trunc, info


_SYNTHETIC = {
    "synthetic": MotionReference.synthetic_walk,
    "synthetic_walk": MotionReference.synthetic_walk,
    "synthetic_spin_attack": MotionReference.synthetic_spin_attack,
    "synthetic_roundhouse_kick": MotionReference.synthetic_roundhouse_kick,
}


def _load_motion(
    path: str | None, fps_override: float | None, device: str
) -> MotionReference:
    """Resolve a motion_path string to a MotionReference instance.

    Sentinel strings (see _SYNTHETIC) select built-in scripted motions; any
    other string is treated as an NPZ path.
    """
    if path is None:
        return MotionReference.synthetic_walk(device=device)
    if path in _SYNTHETIC:
        return _SYNTHETIC[path](device=device)
    return MotionReference.from_npz(path, device=device)
