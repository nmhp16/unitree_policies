"""Stub: SMPL motion (AMASS / WHAM / PHC output) → G1 reference NPZ.

This is the *high-fidelity* front end that goes through SMPL — better quality
than the MediaPipe path but heavier to set up. If you just want to retarget a
YouTube clip and start training, use `retarget_mediapipe.py` instead.

The full SMPL chain has three stages — pick the entry point that matches your
input:

  1) RGB video → SMPL params per frame  (skip if you already have SMPL/AMASS)
     Tools:
       - WHAM       https://github.com/yohanshin/WHAM
       - GVHMR      https://github.com/zju3dv/GVHMR
       - NLF        https://github.com/isarandi/nlf
     Output: a sequence of (root_orient, body_pose, transl, betas) per frame.

  2) SMPL → G1 joint trajectory  (retargeting)
     Tools:
       - PHC / PHC++   https://github.com/ZhengyiLuo/PHC
       - HumanPlus's retarget pipeline (Stanford)
       - Custom: SMPL skeleton → G1 skeleton via joint-correspondence table
     Output: per-frame G1 joint positions (29 DoF), root pose, root velocity.

  3) Save as NPZ matching the schema in envs/motion_reference.py.

When to use this path instead of MediaPipe:
  - You have AMASS/HumanML3D mocap (already SMPL — skips stage 1)
  - You need higher fidelity than monocular video can give
  - You want hand articulation (SMPL-X)
  - You're using a text-to-motion model (MDM, MotionGPT) — those output SMPL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


G1_29DOF_NAMES = (
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


def save_g1_motion_npz(
    output_path: str | Path,
    joint_pos: np.ndarray,        # (T, 29)
    root_pos: np.ndarray,         # (T, 3)
    root_rot: np.ndarray,         # (T, 4) wxyz
    fps: float,
    loopable: bool = False,
    joint_vel: np.ndarray | None = None,
    root_lin_vel: np.ndarray | None = None,
    root_ang_vel: np.ndarray | None = None,
):
    """Pack a retargeted motion into the NPZ schema MotionReference reads."""
    assert joint_pos.shape[1] == 29, f"expected 29 joints, got {joint_pos.shape[1]}"
    out = {
        "fps": np.float32(fps),
        "joint_names": np.array(G1_29DOF_NAMES),
        "joint_pos": joint_pos.astype(np.float32),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "loopable": np.bool_(loopable),
    }
    if joint_vel is not None:
        out["joint_vel"] = joint_vel.astype(np.float32)
    if root_lin_vel is not None:
        out["root_lin_vel"] = root_lin_vel.astype(np.float32)
    if root_ang_vel is not None:
        out["root_ang_vel"] = root_ang_vel.astype(np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file, SMPL pkl, or AMASS npz")
    ap.add_argument("--output", required=True, help="output NPZ path")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    raise NotImplementedError(
        f"Stage 1+2 (video→SMPL→G1 retarget) is not implemented yet.\n"
        f"To enable this pipeline:\n"
        f"  - Install WHAM (https://github.com/yohanshin/WHAM) for stage 1, OR\n"
        f"    use AMASS/HumanML3D directly to skip stage 1.\n"
        f"  - Install PHC++ (https://github.com/ZhengyiLuo/PHC) for stage 2.\n"
        f"  - Then implement load_smpl(args.input) and retarget_to_g1(smpl)\n"
        f"    here, and call save_g1_motion_npz(args.output, ...).\n\n"
        f"For now, use motion.path: synthetic in your task YAML for smoke tests."
    )


if __name__ == "__main__":
    main()
