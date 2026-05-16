"""Reference motion data for DeepMimic-style tracking.

The training pipeline expects per-frame reference state for the G1 (29 DoF).
Real references come from the video → SMPL → G1 retarget pipeline (see
`scripts/convert_smpl_to_npz.py`). A synthetic walking gait is provided so
the env is runnable end-to-end without external motion data — useful for
smoke-testing the RL plumbing before plugging in real data.

NPZ schema (all required unless noted):
    fps           : scalar       — frame rate of the reference
    joint_names   : (J,) str     — names in the order joint_pos columns appear
    joint_pos     : (T, J) f32   — target joint positions per frame
    joint_vel     : (T, J) f32   — target joint velocities (computed if absent)
    root_pos      : (T, 3) f32   — world-frame root position (x, y, z)
    root_rot      : (T, 4) f32   — root orientation (w, x, y, z) quaternion
    root_lin_vel  : (T, 3) f32   — root linear velocity (optional, finite-diff)
    root_ang_vel  : (T, 3) f32   — root angular velocity (optional, finite-diff)
    loopable      : bool         — if True, phase wraps; else episode ends at T
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


_G1_29DOF_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


@dataclass
class MotionData:
    fps: float
    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor          # (T, J)
    joint_vel: torch.Tensor          # (T, J)
    root_pos: torch.Tensor           # (T, 3)
    root_rot: torch.Tensor           # (T, 4) wxyz
    root_lin_vel: torch.Tensor       # (T, 3)
    root_ang_vel: torch.Tensor       # (T, 3)
    loopable: bool

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


class MotionReference:
    """Holds reference motion + queries by continuous phase ∈ [0, 1).

    Phase indexing is fps-independent — the env increments phase by
    `sim_dt / motion_duration` each step, and `query()` linearly interpolates
    between bracketing reference frames.
    """

    def __init__(self, motion: MotionData, device: torch.device | str = "cuda"):
        self.motion = motion
        self.device = torch.device(device)
        # Move all tensors to device once
        self._jp = motion.joint_pos.to(self.device)
        self._jv = motion.joint_vel.to(self.device)
        self._rp = motion.root_pos.to(self.device)
        self._rr = motion.root_rot.to(self.device)
        self._rlv = motion.root_lin_vel.to(self.device)
        self._rav = motion.root_ang_vel.to(self.device)

    @property
    def num_joints(self) -> int:
        return self._jp.shape[1]

    @property
    def num_frames(self) -> int:
        return self._jp.shape[0]

    @property
    def duration(self) -> float:
        return self.motion.duration

    def query(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """Linearly interpolate reference state at the given continuous phase.

        phase: (N,) float in [0, 1).
        Returns dict with batched tensors of shape (N, ...).
        """
        T = self.num_frames
        f = phase.clamp(0.0, 1.0 - 1e-6) * (T - 1)
        lo = f.floor().long()
        hi = (lo + 1).clamp(max=T - 1)
        a = (f - lo.float()).unsqueeze(-1)

        def interp(buf: torch.Tensor) -> torch.Tensor:
            return buf[lo] * (1 - a) + buf[hi] * a

        return {
            "joint_pos": interp(self._jp),
            "joint_vel": interp(self._jv),
            "root_pos": interp(self._rp),
            "root_rot": _slerp(self._rr[lo], self._rr[hi], a.squeeze(-1)),
            "root_lin_vel": interp(self._rlv),
            "root_ang_vel": interp(self._rav),
        }

    def index(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """Nearest-frame lookup (no interpolation) — for RSI / hard snapping.

        Used to snap robot state at episode reset to an exact reference frame.
        Interpolation between frames is fine for reward computation but causes
        slight quaternion/positional inconsistencies that confuse the physics
        engine on a hard write.
        """
        f = (phase.clamp(0.0, 1.0 - 1e-6) * (self.num_frames - 1)).long()
        return {
            "joint_pos": self._jp[f],
            "joint_vel": self._jv[f],
            "root_pos": self._rp[f],
            "root_rot": self._rr[f],
            "root_lin_vel": self._rlv[f],
            "root_ang_vel": self._rav[f],
        }

    @property
    def loopable(self) -> bool:
        return self.motion.loopable

    @classmethod
    def from_npz(cls, path: str | Path, device: str = "cuda") -> "MotionReference":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference motion not found: {path}\n"
                f"Generate one with scripts/convert_smpl_to_npz.py, or use "
                f"MotionReference.synthetic_walk() for a placeholder."
            )
        data = np.load(path, allow_pickle=False)
        return cls(_motion_from_npz(data), device=device)

    @classmethod
    def synthetic_walk(
        cls,
        duration_s: float = 2.0,
        fps: float = 60.0,
        device: str = "cuda",
    ) -> "MotionReference":
        """Cycle-loopable synthetic walk for smoke-testing without real data.

        Produces a believable but not realistic gait: sinusoidal hip pitch and
        knee flexion 180° out of phase between legs, arms swinging counter to
        legs. Root advances at ~0.8 m/s in +x with small vertical bobbing.
        Useful for verifying the tracking RL plumbing end-to-end.
        """
        T = int(duration_s * fps)
        t = torch.linspace(0, duration_s, T)
        phase = 2 * torch.pi * t / duration_s
        names = _G1_29DOF_NAMES
        J = len(names)
        jp = torch.zeros(T, J)

        def col(name: str) -> int:
            return names.index(name)

        amp_hip = 0.45
        amp_knee = 0.55
        amp_ankle = 0.25
        amp_shoulder = 0.35

        # Legs: left leads, right trails by π
        jp[:, col("left_hip_pitch_joint")] = -amp_hip * torch.sin(phase)
        jp[:, col("right_hip_pitch_joint")] = -amp_hip * torch.sin(phase + torch.pi)
        # Knees flex when leg is rear-going (so knee follows after hip)
        jp[:, col("left_knee_joint")] = amp_knee * (0.5 - 0.5 * torch.cos(phase))
        jp[:, col("right_knee_joint")] = amp_knee * (0.5 - 0.5 * torch.cos(phase + torch.pi))
        jp[:, col("left_ankle_pitch_joint")] = -amp_ankle * torch.sin(phase + 0.3)
        jp[:, col("right_ankle_pitch_joint")] = -amp_ankle * torch.sin(phase + torch.pi + 0.3)
        # Arms swing opposite legs
        jp[:, col("left_shoulder_pitch_joint")] = amp_shoulder * torch.sin(phase + torch.pi)
        jp[:, col("right_shoulder_pitch_joint")] = amp_shoulder * torch.sin(phase)
        # Slight elbow flex constant
        jp[:, col("left_elbow_joint")] = 0.3
        jp[:, col("right_elbow_joint")] = 0.3

        joint_vel = torch.zeros_like(jp)
        joint_vel[1:] = (jp[1:] - jp[:-1]) * fps
        joint_vel[0] = joint_vel[1]

        # Root: advance at 0.8 m/s in +x, small z bob
        root_pos = torch.zeros(T, 3)
        root_pos[:, 0] = 0.8 * t
        root_pos[:, 2] = 0.74 + 0.02 * torch.cos(2 * phase)
        root_rot = torch.zeros(T, 4)
        root_rot[:, 0] = 1.0  # identity quat (wxyz)
        root_lin_vel = torch.zeros(T, 3)
        root_lin_vel[:, 0] = 0.8
        root_ang_vel = torch.zeros(T, 3)

        return cls(
            MotionData(
                fps=fps, joint_names=names,
                joint_pos=jp, joint_vel=joint_vel,
                root_pos=root_pos, root_rot=root_rot,
                root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
                loopable=True,
            ),
            device=device,
        )

    @classmethod
    def synthetic_spin_attack(
        cls,
        seconds_per_revolution: float = 2.0,
        fps: float = 60.0,
        shoulder_roll_out: float = 1.5,
        elbow_bend: float = 0.0,
        spin_direction: int = 1,
        device: str = "cuda",
    ) -> "MotionReference":
        """Stylized "spin with arms outstretched" motion.

        - Both arms abducted to ~horizontal via shoulder_roll
        - Root yaws at constant angular velocity around the world Z axis
        - Root position stays at standing height; no forward translation
        - Loopable: one full revolution (phase 0→1) returns to the start pose

        The motion is fully synthetic — no external data. The hardest part for
        the policy isn't matching the pose (easy) but learning to *pivot* on
        flat ground at the commanded angular velocity. That's a contact-rich
        skill that DeepMimic tracking reward is well-suited to discover.

        Args:
            seconds_per_revolution: time for one full 2π yaw revolution
            shoulder_roll_out: target shoulder_roll abduction (rad).
                1.5 ≈ 86° (T-pose-ish). Drop to 1.0 for a less wide stance.
            elbow_bend: elbow joint angle (rad). 0 = straight arms.
            spin_direction: +1 for counter-clockwise (yaw+), -1 for CW.
        """
        T = int(seconds_per_revolution * fps)
        names = _G1_29DOF_NAMES
        J = len(names)
        jp = torch.zeros(T, J)

        def col(name: str) -> int:
            return names.index(name)

        # Stand: slight hip-pitch + knee bend so legs aren't locked straight
        jp[:, col("left_hip_pitch_joint")] = -0.10
        jp[:, col("right_hip_pitch_joint")] = -0.10
        jp[:, col("left_knee_joint")] = 0.30
        jp[:, col("right_knee_joint")] = 0.30
        jp[:, col("left_ankle_pitch_joint")] = -0.20
        jp[:, col("right_ankle_pitch_joint")] = -0.20
        # Arms out — left positive roll, right negative roll (G1 mirror conv.)
        jp[:, col("left_shoulder_roll_joint")] = shoulder_roll_out
        jp[:, col("right_shoulder_roll_joint")] = -shoulder_roll_out
        jp[:, col("left_elbow_joint")] = elbow_bend
        jp[:, col("right_elbow_joint")] = elbow_bend

        joint_vel = torch.zeros_like(jp)  # static pose → zero target joint vel

        # Root: stay in place at standing height
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.74

        # Root yaw: constant angular velocity around world Z
        omega = spin_direction * 2.0 * torch.pi / seconds_per_revolution
        t = torch.linspace(0, seconds_per_revolution, T)
        yaw = omega * t  # wraps naturally with phase loop
        root_rot = torch.zeros(T, 4)
        root_rot[:, 0] = (yaw * 0.5).cos()      # w
        root_rot[:, 3] = (yaw * 0.5).sin()      # z (rotation around Z axis only)

        root_lin_vel = torch.zeros(T, 3)         # no translation
        root_ang_vel = torch.zeros(T, 3)
        root_ang_vel[:, 2] = omega              # constant yaw rate

        return cls(
            MotionData(
                fps=fps, joint_names=names,
                joint_pos=jp, joint_vel=joint_vel,
                root_pos=root_pos, root_rot=root_rot,
                root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
                loopable=True,
            ),
            device=device,
        )

    @classmethod
    def synthetic_roundhouse_kick(
        cls,
        duration_s: float = 1.2,
        fps: float = 60.0,
        kick_leg: str = "right",
        peak_hip_pitch: float = -0.9,
        peak_hip_roll: float = 0.5,
        peak_knee_straight: float = 0.15,
        device: str = "cuda",
    ) -> "MotionReference":
        """One-shot roundhouse kick — harder example to study against the spin.

        Single-leg balance + specific pose + rapid mid-trajectory peak. This
        teaches the two patterns most stylized motions use:

          (a) Non-loopable trajectory. The motion has a start, a peak, and a
              return. loopable=False so the episode ends at phase=1.0 via the
              `motion_completed` termination. RSI samples phase in [0, 0.85)
              so episodes have time to play out before termination fires.

          (b) Mid-motion peak via sin² bump. The kicking leg's joint angles
              move from rest → peak (at phase=0.5) → rest. sin²(π·phase) gives
              a smooth bump that's 0 at the endpoints and 1 in the middle —
              cleaner than two linear ramps because the velocity profile is
              also smooth (zero at endpoints, max in the middle).

        The mechanics the policy has to discover (not supplied):
          - Shift weight onto the standing leg before lifting the other
          - Counter-rotate torso/arms for angular momentum balance
          - Plant the kicking leg cleanly on retract without overshoot
        """
        T = int(duration_s * fps)
        names = _G1_29DOF_NAMES
        J = len(names)
        jp = torch.zeros(T, J)

        def col(name: str) -> int:
            return names.index(name)

        # Base standing pose for both legs (kicking leg starts here too)
        base_hip_pitch = -0.10
        base_knee = 0.30
        base_ankle = -0.20
        for side in ("left", "right"):
            jp[:, col(f"{side}_hip_pitch_joint")] = base_hip_pitch
            jp[:, col(f"{side}_knee_joint")] = base_knee
            jp[:, col(f"{side}_ankle_pitch_joint")] = base_ankle

        # Arms moderately extended for balance — not full T-pose, just out
        jp[:, col("left_shoulder_roll_joint")] = 0.6
        jp[:, col("right_shoulder_roll_joint")] = -0.6
        jp[:, col("left_elbow_joint")] = 0.3
        jp[:, col("right_elbow_joint")] = 0.3

        # sin²(π·phase) bump: peaks at phase=0.5, zero at 0 and 1
        phase = torch.linspace(0, 1, T)
        bump = (torch.pi * phase).sin().pow(2)

        # Kicking leg: deviation from base, weighted by bump
        L = kick_leg
        sign = 1.0 if L == "right" else -1.0
        jp[:, col(f"{L}_hip_pitch_joint")] += bump * (peak_hip_pitch - base_hip_pitch)
        jp[:, col(f"{L}_hip_roll_joint")] += bump * (sign * peak_hip_roll)
        jp[:, col(f"{L}_knee_joint")] += bump * (peak_knee_straight - base_knee)

        # Counter-lean of the waist toward the standing leg for balance
        jp[:, col("waist_roll_joint")] += bump * (-sign * 0.15)
        # Slight torso yaw to load the kick rotation
        jp[:, col("waist_yaw_joint")] += bump * (sign * 0.20)

        # Velocities: finite diff (kick has rapid motion — vel tracking matters)
        joint_vel = torch.zeros_like(jp)
        joint_vel[1:] = (jp[1:] - jp[:-1]) * fps
        joint_vel[0] = joint_vel[1]

        # Root: stays in place, identity orientation. The kicking leg moves,
        # not the body. (Real kicks DO involve torso translation/rotation, but
        # keeping root pose fixed is a deliberate simplification — gives the
        # policy something concrete to anchor to. Loosen later if needed.)
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.74
        root_rot = torch.zeros(T, 4)
        root_rot[:, 0] = 1.0
        root_lin_vel = torch.zeros(T, 3)
        root_ang_vel = torch.zeros(T, 3)

        return cls(
            MotionData(
                fps=fps, joint_names=names,
                joint_pos=jp, joint_vel=joint_vel,
                root_pos=root_pos, root_rot=root_rot,
                root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
                loopable=False,
            ),
            device=device,
        )


def _motion_from_npz(data) -> MotionData:
    fps = float(data["fps"])
    jp = torch.from_numpy(data["joint_pos"]).float()
    jv = (
        torch.from_numpy(data["joint_vel"]).float()
        if "joint_vel" in data.files
        else _finite_diff(jp, fps)
    )
    rp = torch.from_numpy(data["root_pos"]).float()
    rr = torch.from_numpy(data["root_rot"]).float()
    rlv = (
        torch.from_numpy(data["root_lin_vel"]).float()
        if "root_lin_vel" in data.files
        else _finite_diff(rp, fps)
    )
    rav = (
        torch.from_numpy(data["root_ang_vel"]).float()
        if "root_ang_vel" in data.files
        else torch.zeros_like(rp)
    )
    names = tuple(str(n) for n in data["joint_names"])
    loopable = bool(data["loopable"]) if "loopable" in data.files else False
    return MotionData(
        fps=fps, joint_names=names,
        joint_pos=jp, joint_vel=jv,
        root_pos=rp, root_rot=rr,
        root_lin_vel=rlv, root_ang_vel=rav,
        loopable=loopable,
    )


def _finite_diff(x: torch.Tensor, fps: float) -> torch.Tensor:
    v = torch.zeros_like(x)
    v[1:] = (x[1:] - x[:-1]) * fps
    v[0] = v[1]
    return v


def _slerp(qa: torch.Tensor, qb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interp on (..., 4) wxyz quats. t is (...,)."""
    dot = (qa * qb).sum(-1, keepdim=True)
    qb = torch.where(dot < 0, -qb, qb)
    dot = dot.abs().clamp(max=1.0 - 1e-6)
    theta = dot.acos()
    sin_t = theta.sin()
    wa = ((1 - t.unsqueeze(-1)) * theta).sin() / sin_t
    wb = (t.unsqueeze(-1) * theta).sin() / sin_t
    return wa * qa + wb * qb
