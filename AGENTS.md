# AGENTS.md

Guidance for agents working on this codebase. Focused on patterns that help
this kind of pipeline stay maintainable; not exhaustive.

## Project shape

Motion-tracking RL on the Unitree G1. Reference motions live as either
scripted Python functions or NPZ files. A PPO policy is trained to physically
reproduce each clip in Isaac Lab. See README.md for the user-facing flow.

## Before changing anything

Read these three files first — they hold most of the design:

- `envs/motion_reference.py` — reference loader + the worked examples
- `envs/g1_tracking_env_cfg.py` — env + manager scaffolding + defaults
- `envs/mdp.py` — every reward, observation, termination, event function

Most "change a behavior" tasks are actually edits to a YAML in `tasks/`,
not code edits. Check that first.

## Verify by running

There is no unit test suite. The validation path is the smoke test:

    ./scripts/train_rl.sh --task spin_attack --num_envs 64 \
        --max_iterations 2 --headless

If this crashes, the change is broken. Static analysis is not enough — Isaac
Lab's manager system resolves a lot at runtime. Run it.

## Workflow when adding a motion

The pattern that keeps things clean:

1. **Visualize the reference before you train.** If the reference itself is
   wrong, no amount of training fixes it. A `play.py`-style script that
   replays the reference without a policy is the highest-leverage debugging
   tool.
2. **Smoke test with synthetic data first.** Get the RL plumbing working on a
   built-in `synthetic_*` motion before trying retargeted data. Failures
   isolate cleanly that way.
3. **Train short, then long.** Two PPO iterations confirms the pipeline is
   wired. Don't queue a 1-hour run before that succeeds.
4. **Tune one knob at a time.** Reward weights, termination thresholds, and
   domain randomization interact. Change one, observe, then the next.

## Reward engineering

These are the hard-won lessons of motion-tracking RL. Re-derive at your own
cost.

- **Reference State Initialization (RSI) is non-negotiable.** Reset to a
  random phase, not always phase 0. Otherwise the policy bottlenecks on the
  easy beginning of the clip and never sees the hard middle.
- **Early-terminate diverged trajectories.** Without `deviated_from_reference`
  (or equivalent), the policy spends most of its rollouts on already-doomed
  states. Sample efficiency drops 5-10×.
- **Dense reward gets you off the ground; sparse reward keeps you honest.**
  Start with dense tracking; if you have a clean success predicate, schedule
  the dense weights toward 0 over training so the final policy is aligned
  with the sparse goal.
- **Up-weight the signature, zero out the irrelevant.** For each motion ask:
  what is it visually about (joint pose? root rotation? translation?). Bump
  that term. For terms whose reference value is zero (e.g. `root_lin_vel`
  for an in-place spin), set weight to 0 — tracking zero adds noise, not
  signal.
- **Gate dense terms behind their preconditions.** Rewarding "low z" only
  makes sense once "centered xy" is met, else the policy crashes the plant
  through the table instead of into the vial. Gate accordingly.
- **Log every reward component separately.** A single scalar reward hides
  which term is doing the work. rsl_rl logs per-term means by default — use
  them.

## Isaac Lab patterns that bite

- **`AppLauncher` must run before any `from isaaclab.*` import.** Use the
  argparse-then-`AppLauncher`-then-import pattern shown in `scripts/train_rl.py`.
- **MDP functions are stateless.** They take `(env, ...) -> Tensor`. Don't
  stash state on them; stash it on the env.
- **Cache per-step queries.** If multiple manager terms need the same data
  (e.g., the reference at the current phase), cache it on the env keyed by
  `common_step_counter`. See `query_reference()` for the pattern.
- **`SceneEntityCfg("robot")` as default arg.** Don't hardcode `"robot"`
  paths inside functions; accept it as a parameter with this default.
- **`enable_cameras=True` is required for off-screen render.** Headless
  recording silently produces zero-byte mp4s without it. `play.py` forces it
  when `--video` is on.

## Domain randomization (DR) discipline

DR is the difference between sim-only and sim-to-real, but it actively hurts
early training:

- **Start with no DR or very light DR.** Get the policy to match the
  reference cleanly first.
- **Add randomization once the baseline converges.** Friction is the cheapest
  win, then mass, then actuator gains, then sensor latency.
- **Verify each addition independently.** Adding three randomizers at once
  and seeing reward drop tells you nothing about which one broke things.

## When a training run looks wrong

In likely order:

1. **Plot the reference.** Is the motion data what you think it is? Joint
   sign conventions, axis swaps, and unit confusions cause more bugs than
   any reward-shaping mistake.
2. **Check the reward components.** If `track_joint_pos` is high but the
   robot is on the floor, the alive bonus / termination are mis-weighted.
3. **Check the episode length.** Diverged-too-fast or surviving-without-
   tracking are both diagnosable from mean episode length alone.
4. **Run one env, GUI mode, render at full fps.** Watch the policy. A 30-sec
   visual catches what 30 minutes of TensorBoard staring won't.

## Code conventions

- Type hints on public functions; optional on internal helpers.
- Docstrings only when the WHY is non-obvious. Skip them for one-line helpers.
- Comments explain reasoning, not what the code does. If the comment restates
  the code, delete the comment.
- Reward weights belong in YAML, not in code. Don't hardcode.
- Each new motion gets its own task YAML and its own classmethod / NPZ.
  Don't fork the env config.
- Keep `mdp.py` functions side-effect-free. The event manager writes state;
  rewards and observations only read.

## Environments

Two separate pip environments live alongside this project:

- **Isaac Lab's bundled python** — used by `train_rl.sh` and `play.sh` via
  `isaaclab.sh -p`. Carries torch, rsl_rl, `isaaclab_*`, gymnasium.
- **`.venv`** (project-local) — for `retarget_mediapipe.py` only. Has
  mediapipe, opencv, scipy, yt-dlp. See `requirements.txt`.

Don't merge them. MediaPipe pins protobuf and numpy ranges that clash with
Isaac Lab's environment.
