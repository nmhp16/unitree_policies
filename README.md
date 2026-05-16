# unitree-policies

Train Unitree G1 (29 DoF) policies to reproduce motion clips. Motions are
written as short Python functions; PPO learns to physically execute them in
Isaac Lab. A video-retargeting path exists as an escape hatch for motions you
can't script easily, but most useful motions can just be parameterized.

## Setup

Assumes Isaac Lab at `~/IsaacLab` (override with `ISAACLAB=...`). rsl_rl
ships with it. Nothing else needed for training.

## Train

Tasks: `track_walk`, `spin_attack`, `roundhouse_kick`.

    # smoke test, ~2 min
    ./scripts/train_rl.sh --task spin_attack --num_envs 64 --max_iterations 2 --headless

    # real run, ~1-2h on a 4090
    ./scripts/train_rl.sh --task spin_attack --num_envs 4096 --headless

Checkpoints land in `outputs/rl_runs/<task>/`.

## Eval

    ./scripts/play.sh --task spin_attack --checkpoint outputs/rl_runs/spin_attack/final.pt

Writes an mp4 to `outputs/rl_runs/<task>/videos/`. Needs `ffmpeg` on PATH.
Pass `--no-video` for GUI-only.

## Add a new motion

Most motions can be parameterized in a few lines. The pattern:

1. Add a `synthetic_<name>()` classmethod to `MotionReference` in
   `envs/motion_reference.py`. Return a `MotionData` with per-frame joint
   angles + root pose. See `synthetic_spin_attack` or
   `synthetic_roundhouse_kick` as worked examples — most motions are either
   "static pose with body moving" (spin, walk) or "body still with joints
   moving" (kick, throw).
2. Register the sentinel string in `envs/g1_tracking_env._load_motion`.
3. Drop a task YAML in `tasks/` (copy one of the existing ones, change
   `motion.path` to your sentinel, tune reward weights).
4. Train.

## Layout

`envs/` env + rewards + motion loader. `tasks/` YAMLs. `scripts/` entry
points. `configs/` PPO hyperparams. `data/reference/` retargeted NPZs.

## Advanced: train on a real video

For motions that are too complex or stylized to script, retarget a video
through MediaPipe Pose. Needs its own venv since MediaPipe and Isaac Lab
clash on numpy / protobuf versions:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    mkdir -p data/models
    wget -O data/models/pose_landmarker_heavy.task \
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

Then:

    # YouTube search:
    python scripts/retarget_mediapipe.py \
        --search "person spinning with arms out" \
        --output data/reference/spin.npz --loopable

    # Specific URL:     --url "https://www.youtube.com/watch?v=..."
    # Local file:       --video path/to/clip.mp4

Point a task YAML at the resulting NPZ (`motion.path: data/reference/spin.npz`)
and train as usual.

Caveats: MediaPipe depth is noisy on rotational motions, so retargeted
motions often need joint-sign or filtering tweaks before they train cleanly.
For most use cases, scripting the motion directly is faster and more reliable.

## Caveats

Domain randomization is light — meant for sim-only first. The video pipeline
is built but unbattle-tested. Wrists stay at default since neither path
gives reliable hand pose.
