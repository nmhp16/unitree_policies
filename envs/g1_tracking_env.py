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
        # Initialize motion + phase + cache BEFORE super().__init__() so the
        # observation manager can probe term shapes (it calls every obs func
        # once during setup, which needs env.motion_phase + query_reference()).
        self.motion = _load_motion(
            cfg.motion_path, cfg.motion_fps_override, device=cfg.sim.device
        )
        self.motion_phase = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self._ref_cache: dict[str, torch.Tensor] | None = None
        self._ref_cache_step = -1
        # Action-latency state. cfg.action_latency_max_steps sets the maximum
        # control-step delay each env can be assigned (0 disables latency
        # entirely — the default for backward compatibility). The buffer
        # itself is lazy-init'd on the first step() because the action shape
        # isn't known until then.
        self._action_latency_max = int(getattr(cfg, "action_latency_max_steps", 0))
        self._action_buffer: torch.Tensor | None = None
        self._action_buffer_idx = 0
        self._action_latency_per_env: torch.Tensor | None = None
        super().__init__(cfg=cfg, **kwargs)

    @property
    def step_dt(self) -> float:
        return float(self.cfg.sim.dt * self.cfg.decimation)

    def query_reference(self) -> dict[str, torch.Tensor]:
        """Per-step cached reference query — all manager terms see the same state."""
        # common_step_counter is only set after super().__init__(); during the
        # manager's shape-probing pass it doesn't exist yet, so fall back to -1.
        step = getattr(self, "common_step_counter", -1)
        if self._ref_cache_step != step:
            self._ref_cache = self.motion.query(self.motion_phase)
            self._ref_cache_step = step
        return self._ref_cache

    def _apply_action_latency(self, action: torch.Tensor) -> torch.Tensor:
        """Return the delayed action that should actually drive the sim.

        Maintains a circular buffer of past actions and gathers each env's
        action from `latency_per_env` steps ago. Latency is sampled once per
        env at startup (and re-sampled on episode reset). Standard sim-to-real
        trick — the policy learns to handle any actuator delay up to
        `_action_latency_max` control steps.
        """
        if self._action_latency_max <= 0:
            return action
        N = action.shape[0]
        buf_size = self._action_latency_max + 1
        if self._action_buffer is None:
            self._action_buffer = torch.zeros(
                buf_size, *action.shape, device=action.device, dtype=action.dtype
            )
            self._action_latency_per_env = torch.randint(
                0, buf_size, (N,), device=action.device,
            )
        # Push current action to head, then gather delayed slot per env.
        self._action_buffer[self._action_buffer_idx] = action
        delayed_idx = (self._action_buffer_idx - self._action_latency_per_env) % buf_size
        env_ids = torch.arange(N, device=action.device)
        delayed = self._action_buffer[delayed_idx, env_ids]
        self._action_buffer_idx = (self._action_buffer_idx + 1) % buf_size
        return delayed

    def step(self, action):
        action = self._apply_action_latency(action)
        obs, rew, term, trunc, info = super().step(action)
        mdp.advance_motion_phase(self)
        # On episode reset, clear the prior actions out of the buffer (else
        # the next episode's first `latency` steps would replay the dead
        # episode's commands) and re-sample latency per env so each rollout
        # sees a fresh draw from the latency distribution.
        if self._action_latency_max > 0 and self._action_buffer is not None:
            dones = term | trunc
            if dones.any():
                reset_ids = torch.where(dones)[0]
                self._action_buffer[:, reset_ids, :] = 0.0
                self._action_latency_per_env[reset_ids] = torch.randint(
                    0, self._action_latency_max + 1, (len(reset_ids),),
                    device=action.device,
                )
        return obs, rew, term, trunc, info


_SYNTHETIC = {
    "synthetic": MotionReference.synthetic_walk,
    "synthetic_walk": MotionReference.synthetic_walk,
    "synthetic_stand_armsout": MotionReference.synthetic_stand_armsout,
    "synthetic_karate_punch": MotionReference.synthetic_karate_punch,
    "synthetic_wing_chun": MotionReference.synthetic_wing_chun,
    "synthetic_wing_chun_demo": MotionReference.synthetic_wing_chun_demo,
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
