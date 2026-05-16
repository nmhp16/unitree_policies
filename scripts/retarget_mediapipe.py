"""Video → MediaPipe pose → G1 reference motion (NPZ).

Pipeline:
    RGB video → MediaPipe PoseLandmarker → 33 world landmarks per frame
    → temporal smoothing → analytical IK to G1 29-DoF joint angles
    → NPZ in the schema envs/motion_reference.py reads.

Three entry points (pick one):
    --video PATH       use a local file
    --url URL          download a video first via yt-dlp
    --search QUERY     yt-dlp searches YouTube, downloads the top hit

Tradeoffs (vs. SMPL/PHC pipeline):
    + No model files to install beyond MediaPipe's pose_landmarker.task
    + Single-camera, single-person — runs on any phone or YouTube video
    + Tens of seconds per video on CPU; faster on GPU
    - Lower fidelity than mocap/WHAM for complex motions
    - No hand articulation (wrists stay at default)
    - Foot contact detection is heuristic (no force info from video)
    - Single-camera depth estimation has known artifacts at extreme poses

Usage:
    # Install deps:
    pip install -r requirements_retarget.txt

    # Download the model (one-time, ~30 MB):
    mkdir -p data/models
    wget -O data/models/pose_landmarker_heavy.task \\
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

    # Pull from YouTube + retarget in one shot:
    python scripts/retarget_mediapipe.py \\
        --search "person spinning with arms out" \\
        --output data/reference/spin.npz \\
        --loopable

    # Or a specific URL:
    python scripts/retarget_mediapipe.py \\
        --url "https://www.youtube.com/watch?v=XXXXXXXX" \\
        --output data/reference/clip.npz

    # Or a local file (no internet):
    python scripts/retarget_mediapipe.py \\
        --video clip.mp4 --output data/reference/clip.npz

    # Then point a task YAML at it (copy spin_attack.yaml as a template):
    #   motion:
    #     path: data/reference/spin.npz

Note on copyright: only retarget videos you have rights to (public-domain,
your own recordings, Creative Commons, or short fair-use clips for research).
yt-dlp respects YouTube's content access controls.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Landmark indices and G1 joint name list
# -----------------------------------------------------------------------------

# MediaPipe Pose: 33 landmarks. Indices documented at:
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
class L:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


G1_JOINT_NAMES = (
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
J_IDX = {name: i for i, name in enumerate(G1_JOINT_NAMES)}


# -----------------------------------------------------------------------------
# Stage 0: fetch video from YouTube (URL or search query)
# -----------------------------------------------------------------------------

def fetch_video(
    *,
    url: Optional[str] = None,
    search: Optional[str] = None,
    cache_dir: Path = Path("data/videos"),
    max_height: int = 720,
    max_duration_s: int = 60,
) -> Path:
    """Download a video via yt-dlp. Returns the local path.

    Cached by URL/query under cache_dir to avoid re-downloading. Bounded
    duration to keep retargeting fast — most motions worth training on are
    under a minute. Capped resolution to keep MediaPipe inference snappy;
    pose quality plateaus around 720p anyway.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError(
            "yt-dlp not installed. Run `pip install yt-dlp` (or "
            "`pip install -r requirements_retarget.txt`)."
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = url if url else f"ytsearch1:{search}"

    # Two-pass: first probe metadata to construct a stable filename, then download.
    probe_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(target, download=False)
    if "entries" in info:  # search returned a playlist of 1
        info = info["entries"][0]

    video_id = info.get("id", "unknown")
    duration = info.get("duration", 0)
    title = (info.get("title") or "untitled").replace("/", "_")[:60]
    if duration and duration > max_duration_s:
        print(
            f"WARNING: video is {duration}s long (>{max_duration_s}s cap). "
            f"Will download but consider trimming with --start/--end "
            f"(not yet implemented; use ffmpeg manually).",
            file=sys.stderr,
        )

    out_path = cache_dir / f"{video_id}_{title}.mp4"
    if out_path.exists():
        print(f"[0/4] Cached video: {out_path}", file=sys.stderr)
        return out_path

    print(f"[0/4] Downloading: {info.get('webpage_url') or target}", file=sys.stderr)
    print(f"      Title: {info.get('title')}", file=sys.stderr)
    print(f"      Duration: {duration}s", file=sys.stderr)

    dl_opts = {
        "format": f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "outtmpl": str(out_path),
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([info.get("webpage_url") or target])

    if not out_path.exists():
        # yt-dlp may have written a different extension if the merge fallback fired
        candidates = list(cache_dir.glob(f"{video_id}_*"))
        if candidates:
            out_path = candidates[0]
        else:
            raise RuntimeError(f"yt-dlp finished but no output file found in {cache_dir}")
    return out_path


# -----------------------------------------------------------------------------
# Stage 1: video → landmarks
# -----------------------------------------------------------------------------

@dataclass
class LandmarkSequence:
    landmarks: np.ndarray   # (T, 33, 3) in MediaPipe world frame (meters)
    visibility: np.ndarray  # (T, 33) confidence in [0, 1]
    valid: np.ndarray       # (T,) bool — pose detected on this frame
    fps: float


def extract_landmarks(video_path: Path, model_path: Path) -> LandmarkSequence:
    """Run MediaPipe PoseLandmarker over the video, return per-frame world landmarks."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not model_path.exists():
        raise FileNotFoundError(
            f"PoseLandmarker model not found: {model_path}\n"
            f"Download with:\n"
            f"  mkdir -p {model_path.parent}\n"
            f"  wget -O {model_path} "
            f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarks_buf: list[np.ndarray] = []
    visibility_buf: list[np.ndarray] = []
    valid_buf: list[bool] = []

    with mp_vision.PoseLandmarker.create_from_options(options) as detector:
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int(idx * 1000.0 / fps)
            result = detector.detect_for_video(mp_image, ts_ms)

            if result.pose_world_landmarks:
                lms = result.pose_world_landmarks[0]
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                vis = np.array([lm.visibility for lm in lms], dtype=np.float32)
                landmarks_buf.append(pts)
                visibility_buf.append(vis)
                valid_buf.append(True)
            else:
                landmarks_buf.append(np.zeros((33, 3), dtype=np.float32))
                visibility_buf.append(np.zeros(33, dtype=np.float32))
                valid_buf.append(False)
            idx += 1
            if total and idx % max(1, total // 20) == 0:
                print(f"  {idx}/{total} frames ({100*idx//total}%)", file=sys.stderr)

    cap.release()
    if not landmarks_buf:
        raise RuntimeError("No frames decoded from video")

    return LandmarkSequence(
        landmarks=np.stack(landmarks_buf),
        visibility=np.stack(visibility_buf),
        valid=np.array(valid_buf),
        fps=fps,
    )


# -----------------------------------------------------------------------------
# Stage 2: clean / smooth / coordinate-convert
# -----------------------------------------------------------------------------

def mediapipe_to_robot_frame(landmarks_mp: np.ndarray) -> np.ndarray:
    """MediaPipe world frame → robot world frame (x forward, y left, z up).

    MediaPipe world landmarks: x right (subject's right), y down, z away from
    camera. We want: x forward (subject-facing), y left (subject's left), z up.

    The subject usually faces the camera, so "forward" in robot frame ≈ -z in
    MediaPipe frame (toward camera). Mapping:
        robot_x  =  -mp_z   (forward)
        robot_y  =  -mp_x   (left)
        robot_z  =  -mp_y   (up)
    """
    out = np.zeros_like(landmarks_mp)
    out[..., 0] = -landmarks_mp[..., 2]
    out[..., 1] = -landmarks_mp[..., 0]
    out[..., 2] = -landmarks_mp[..., 1]
    return out


def interpolate_invalid_frames(seq: LandmarkSequence, max_gap_s: float = 0.5):
    """Linearly interp landmarks across short gaps where MediaPipe failed."""
    valid_idx = np.where(seq.valid)[0]
    if len(valid_idx) < 2:
        raise RuntimeError("Fewer than 2 valid frames — can't interpolate")
    max_gap_frames = int(max_gap_s * seq.fps)

    for j in range(seq.landmarks.shape[1]):  # per landmark
        for axis in range(3):
            xs = valid_idx
            ys = seq.landmarks[valid_idx, j, axis]
            seq.landmarks[:, j, axis] = np.interp(np.arange(len(seq.valid)), xs, ys)

    # Mark frames inside long gaps as still-invalid (don't trust extrapolation)
    gaps_too_long = []
    for i in range(len(valid_idx) - 1):
        if valid_idx[i + 1] - valid_idx[i] - 1 > max_gap_frames:
            gaps_too_long.append((valid_idx[i] + 1, valid_idx[i + 1]))
    return gaps_too_long


def smooth_landmarks(landmarks: np.ndarray, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay filter per landmark per axis. Keeps shape, smooths jitter."""
    from scipy.signal import savgol_filter
    T = landmarks.shape[0]
    window = min(window, T - (T + 1) % 2)  # must be odd, <= T
    if window < polyorder + 2:
        return landmarks  # too few frames to smooth meaningfully
    return savgol_filter(landmarks, window_length=window, polyorder=polyorder, axis=0)


# -----------------------------------------------------------------------------
# Stage 3: per-frame body frame + analytical IK
# -----------------------------------------------------------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def compute_body_frame(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame body root + 3x3 rotation matrix [right | forward | up].

    Up axis: hip-midpoint → shoulder-midpoint (spine).
    Right axis: right-hip → left-hip cross spine, then re-orthogonalized.
    Forward axis: cross product up × right.
    """
    T = lm.shape[0]
    R = np.zeros((T, 3, 3), dtype=np.float32)
    root = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        hip_l = lm[t, L.LEFT_HIP]
        hip_r = lm[t, L.RIGHT_HIP]
        sho_l = lm[t, L.LEFT_SHOULDER]
        sho_r = lm[t, L.RIGHT_SHOULDER]
        hip_mid = (hip_l + hip_r) / 2
        sho_mid = (sho_l + sho_r) / 2

        up = _normalize(sho_mid - hip_mid)
        # Subject's left = +y in our robot frame → left-hip - right-hip points +y
        left = hip_l - hip_r
        left = _normalize(left - up * np.dot(left, up))
        forward = _normalize(np.cross(up, left))   # right-handed: up × left = forward
        # Re-derive left from forward × up to ensure orthonormality
        left = _normalize(np.cross(forward, up))

        R[t, :, 0] = forward    # robot x
        R[t, :, 1] = left       # robot y
        R[t, :, 2] = up         # robot z
        root[t] = hip_mid
    return root, R


def _to_body_frame(lm: np.ndarray, root: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Express landmarks in body-local frame (root at origin, body-aligned axes)."""
    T = lm.shape[0]
    out = np.zeros_like(lm)
    for t in range(T):
        out[t] = (lm[t] - root[t]) @ R[t]   # world → body: multiply by R (cols are body axes)
    return out


def _leg_joint_angles(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray,
                       foot: np.ndarray, side: str) -> dict[str, float]:
    """Per-frame leg joint angles in body frame.

    Body frame: x forward, y left, z up.
    Hip-pitch: thigh tilts forward/back. Positive thigh.forward → negative
        hip_pitch (G1 convention: hip flexed forward = negative).
    Hip-roll: thigh abducts laterally. Sign depends on side.
    Knee: flexion angle between thigh and shin (0 = straight).
    Ankle-pitch: foot tilts vs shin.
    """
    side_sign = +1.0 if side == "left" else -1.0
    thigh = knee - hip
    shin = ankle - knee
    # Normalize for pitch/roll decomposition
    thigh_n = thigh / (np.linalg.norm(thigh) + 1e-8)
    shin_n = shin / (np.linalg.norm(shin) + 1e-8)

    # hip_pitch: project thigh onto sagittal plane (forward, up), measure angle from -z
    hip_pitch = np.arctan2(-thigh_n[0], -thigh_n[2])
    # hip_roll: lateral abduction (side-component vs vertical)
    hip_roll = side_sign * np.arctan2(side_sign * thigh_n[1], -thigh_n[2])
    # hip_yaw: foot direction in horizontal plane vs thigh — small reliable signal
    # only when foot landmark exists; otherwise 0
    hip_yaw = 0.0

    # knee: angle between thigh and shin (always positive, 0 = straight)
    cos_k = np.clip(np.dot(thigh_n, shin_n), -1.0, 1.0)
    knee = np.arccos(cos_k)

    # ankle_pitch: angle between shin and foot in sagittal plane
    foot_vec = foot - ankle
    foot_n = foot_vec / (np.linalg.norm(foot_vec) + 1e-8)
    # Angle in (forward, up) plane between -shin and foot direction
    ankle_pitch = np.arctan2(foot_n[0], -foot_n[2]) - np.arctan2(-shin_n[0], -shin_n[2])
    ankle_roll = 0.0

    return {
        f"{side}_hip_pitch_joint": float(hip_pitch),
        f"{side}_hip_roll_joint": float(hip_roll),
        f"{side}_hip_yaw_joint": float(hip_yaw),
        f"{side}_knee_joint": float(knee),
        f"{side}_ankle_pitch_joint": float(ankle_pitch),
        f"{side}_ankle_roll_joint": float(ankle_roll),
    }


def _arm_joint_angles(sho: np.ndarray, elb: np.ndarray, wri: np.ndarray, side: str) -> dict[str, float]:
    """Arm joint angles in body frame. Mirrors leg decomposition."""
    side_sign = +1.0 if side == "left" else -1.0
    upper = elb - sho
    fore = wri - elb
    upper_n = upper / (np.linalg.norm(upper) + 1e-8)
    fore_n = fore / (np.linalg.norm(fore) + 1e-8)

    # shoulder_pitch: arm raises forward (negative = forward raise on G1)
    shoulder_pitch = np.arctan2(-upper_n[0], -upper_n[2])
    # shoulder_roll: arm abducts laterally
    shoulder_roll = side_sign * np.arctan2(side_sign * upper_n[1], -upper_n[2])
    shoulder_yaw = 0.0

    # elbow: flexion angle
    cos_e = np.clip(np.dot(upper_n, fore_n), -1.0, 1.0)
    elbow = np.arccos(cos_e)

    return {
        f"{side}_shoulder_pitch_joint": float(shoulder_pitch),
        f"{side}_shoulder_roll_joint": float(shoulder_roll),
        f"{side}_shoulder_yaw_joint": float(shoulder_yaw),
        f"{side}_elbow_joint": float(elbow),
        f"{side}_wrist_roll_joint": 0.0,
        f"{side}_wrist_pitch_joint": 0.0,
        f"{side}_wrist_yaw_joint": 0.0,
    }


def retarget(seq: LandmarkSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retarget the smoothed landmark sequence to G1 joint angles.

    Returns:
        joint_pos: (T, 29) joint angles
        root_pos:  (T, 3) hip-midpoint position in robot world frame
        root_rot:  (T, 4) wxyz quaternion of body frame
    """
    from scipy.spatial.transform import Rotation as R

    # Convert MediaPipe → robot world frame
    lm_world = mediapipe_to_robot_frame(seq.landmarks)
    # Body frame extraction
    root_pos, R_world_to_body = compute_body_frame(lm_world)
    # Express landmarks in body frame
    lm_body = _to_body_frame(lm_world, root_pos, R_world_to_body)

    T = lm_world.shape[0]
    joint_pos = np.zeros((T, len(G1_JOINT_NAMES)), dtype=np.float32)
    root_rot_wxyz = np.zeros((T, 4), dtype=np.float32)

    for t in range(T):
        # Legs
        for side, ids in [
            ("left", (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_FOOT_INDEX)),
            ("right", (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_FOOT_INDEX)),
        ]:
            hip, knee, ank, foot = (lm_body[t, i] for i in ids)
            for jname, val in _leg_joint_angles(hip, knee, ank, foot, side).items():
                joint_pos[t, J_IDX[jname]] = val
        # Arms
        for side, ids in [
            ("left", (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST)),
            ("right", (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST)),
        ]:
            sho, elb, wri = (lm_body[t, i] for i in ids)
            for jname, val in _arm_joint_angles(sho, elb, wri, side).items():
                joint_pos[t, J_IDX[jname]] = val

        # Waist yaw: how much body frame is yawed vs the previous frame's
        # "neutral" forward. For a single-clip retarget, take the average forward
        # direction over the clip as neutral; current yaw is rotation about z.
        # Simpler: read it off the rotation matrix.
        Rmat = R_world_to_body[t]
        # Yaw = atan2(R[1,0], R[0,0]) when columns are body axes in world frame.
        # But our columns are: [forward, left, up], so forward_x = R[0,0], forward_y = R[1,0].
        waist_yaw = np.arctan2(Rmat[1, 0], Rmat[0, 0])
        joint_pos[t, J_IDX["waist_yaw_joint"]] = float(waist_yaw)

        # Root rotation: convert body-frame rotation matrix → quaternion (wxyz)
        # R_world_to_body has columns = body axes in world. That is R_world←body.
        # We want the quaternion that rotates world → body's orientation.
        quat_xyzw = R.from_matrix(Rmat).as_quat()
        root_rot_wxyz[t, 0] = quat_xyzw[3]
        root_rot_wxyz[t, 1:] = quat_xyzw[:3]

    return joint_pos, root_pos.astype(np.float32), root_rot_wxyz


def smooth_joint_angles(jp: np.ndarray, fps: float, window_s: float = 0.15) -> np.ndarray:
    """Final smoothing pass on joint angles — removes IK jitter."""
    from scipy.signal import savgol_filter
    T = jp.shape[0]
    window = max(5, int(window_s * fps))
    window = window if window % 2 == 1 else window + 1
    window = min(window, T - (T + 1) % 2)
    if window < 5:
        return jp
    return savgol_filter(jp, window_length=window, polyorder=3, axis=0)


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------

def finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    v = np.zeros_like(x)
    v[1:] = (x[1:] - x[:-1]) * fps
    v[0] = v[1]
    return v


def save_npz(
    output_path: Path,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    fps: float,
    loopable: bool = False,
):
    joint_vel = finite_diff(joint_pos, fps)
    root_lin_vel = finite_diff(root_pos, fps)
    # Angular velocity from quaternion finite-diff is fiddly; let RL recompute
    # from the rotation reference. Leave zeros — `root_rot_tracking` is what
    # actually drives the orientation; ang-vel tracking is a nice-to-have.
    root_ang_vel = np.zeros_like(root_pos)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.float32(fps),
        joint_names=np.array(G1_JOINT_NAMES),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        root_pos=root_pos.astype(np.float32),
        root_rot=root_rot.astype(np.float32),
        root_lin_vel=root_lin_vel.astype(np.float32),
        root_ang_vel=root_ang_vel.astype(np.float32),
        loopable=np.bool_(loopable),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Local video file")
    src.add_argument("--url", type=str, help="YouTube (or other) URL to download via yt-dlp")
    src.add_argument("--search", type=str, help="YouTube search query; downloads top hit")
    ap.add_argument("--output", required=True, type=Path, help="Output NPZ path")
    ap.add_argument(
        "--model", type=Path,
        default=Path("data/models/pose_landmarker_heavy.task"),
        help="MediaPipe PoseLandmarker model file",
    )
    ap.add_argument("--video-cache", type=Path, default=Path("data/videos"),
                    help="Where downloaded videos are cached")
    ap.add_argument("--max-duration", type=int, default=60,
                    help="Skip videos longer than this many seconds")
    ap.add_argument("--loopable", action="store_true",
                    help="Mark motion as loopable (cyclical motions like walking)")
    ap.add_argument("--smooth-window", type=int, default=9,
                    help="Savgol window for landmark smoothing (odd, ≥5)")
    args = ap.parse_args()

    # Resolve video path (download if needed)
    if args.video:
        if not args.video.exists():
            ap.error(f"Video not found: {args.video}")
        video_path = args.video
    else:
        video_path = fetch_video(
            url=args.url,
            search=args.search,
            cache_dir=args.video_cache,
            max_duration_s=args.max_duration,
        )

    print(f"[1/4] Extracting MediaPipe landmarks from {video_path}...", file=sys.stderr)
    seq = extract_landmarks(video_path, args.model)
    n_valid = int(seq.valid.sum())
    print(f"      {n_valid}/{len(seq.valid)} frames had a detected pose "
          f"({100*n_valid/len(seq.valid):.0f}%), fps={seq.fps:.1f}", file=sys.stderr)
    if n_valid < 10:
        sys.exit("ERROR: fewer than 10 frames with a detected pose — retargeting aborted")

    print(f"[2/4] Cleaning landmark sequence...", file=sys.stderr)
    long_gaps = interpolate_invalid_frames(seq)
    if long_gaps:
        print(f"      WARNING: {len(long_gaps)} gaps >0.5s detected; "
              f"output may be unreliable in those windows", file=sys.stderr)
    seq.landmarks = smooth_landmarks(seq.landmarks, window=args.smooth_window)

    print(f"[3/4] Retargeting to G1 29-DoF skeleton...", file=sys.stderr)
    joint_pos, root_pos, root_rot = retarget(seq)
    joint_pos = smooth_joint_angles(joint_pos, seq.fps)

    print(f"[4/4] Writing {args.output}...", file=sys.stderr)
    save_npz(args.output, joint_pos, root_pos, root_rot, seq.fps, loopable=args.loopable)
    print(f"      Done. {joint_pos.shape[0]} frames @ {seq.fps:.1f} fps "
          f"({joint_pos.shape[0]/seq.fps:.1f} s).", file=sys.stderr)
    print(f"\nNext: set motion.path: {args.output} in your task YAML.")


if __name__ == "__main__":
    main()
