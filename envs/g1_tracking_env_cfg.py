"""Isaac Lab env config for G1 motion tracking (DeepMimic-style).

The full task spec is split across two files:
  - this file: scene + manager scaffolding + sensible defaults
  - tasks/<name>.yaml: per-clip overrides (motion path, reward weights,
    termination thresholds, episode length)

The YAML is overlaid in scripts/train_rl.py via tasks.apply_task_yaml() so
hyperparameter sweeps can hit one file without touching code.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from . import mdp
from .motion_reference import _G1_29DOF_NAMES


# The G1 USD in Isaac Lab includes hand-finger joints (43 DoF total). Our
# reference motion only covers the 29 body joints. Filter everything that
# touches per-joint data — actions, joint obs, tracking rewards, RSI reset,
# divergence termination — to this list. Hand fingers stay at default pose.
_REF_JOINT_NAMES = list(_G1_29DOF_NAMES)
_ROBOT_29DOF = SceneEntityCfg("robot", joint_names=_REF_JOINT_NAMES)


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

@configclass
class G1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0),
    )
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_REF_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": _ROBOT_29DOF},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": _ROBOT_29DOF},
        )
        actions = ObsTerm(func=base_mdp.last_action)
        # Motion conditioning
        phase = ObsTerm(func=mdp.motion_phase)
        reference = ObsTerm(func=mdp.motion_reference_target)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Rewards — DeepMimic weights, tuned via YAML
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # Tracking (positive). All tracking terms see the same 29-joint view as
    # the reference motion via the _ROBOT_29DOF entity cfg.
    track_joint_pos = RewTerm(
        func=mdp.joint_pos_tracking, weight=3.0,
        params={"sigma": 0.25, "asset_cfg": _ROBOT_29DOF},
    )
    track_joint_vel = RewTerm(
        func=mdp.joint_vel_tracking, weight=1.0,
        params={"sigma": 5.0, "asset_cfg": _ROBOT_29DOF},
    )
    track_root_pos = RewTerm(
        func=mdp.root_pos_tracking, weight=2.0, params={"sigma": 0.5}
    )
    track_root_rot = RewTerm(
        func=mdp.root_rot_tracking, weight=1.0, params={"sigma": 0.3}
    )
    track_root_lin_vel = RewTerm(
        func=mdp.root_lin_vel_tracking, weight=1.0, params={"sigma": 1.0}
    )
    track_root_ang_vel = RewTerm(
        func=mdp.root_ang_vel_tracking, weight=1.0, params={"sigma": 1.5}
    )
    alive = RewTerm(func=mdp.alive, weight=0.5)
    upright = RewTerm(func=mdp.upright, weight=0.0, params={"sigma": 0.3})

    # Regularization (negative).
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    torques = RewTerm(
        func=base_mdp.joint_torques_l2, weight=-1.0e-6,
        params={"asset_cfg": _ROBOT_29DOF},
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    fell = DoneTerm(func=mdp.root_too_low, params={"minimum_height": 0.3})
    diverged = DoneTerm(
        func=mdp.deviated_from_reference,
        params={
            "joint_threshold": 1.2, "root_threshold": 0.6,
            "asset_cfg": _ROBOT_29DOF,
        },
    )
    finished = DoneTerm(func=mdp.motion_completed, time_out=True)


# -----------------------------------------------------------------------------
# Events — RSI on reset, domain randomization
# -----------------------------------------------------------------------------

@configclass
class EventsCfg:
    # Reset
    reset_to_ref = EventTerm(
        func=mdp.reset_to_motion_phase, mode="reset",
        params={"asset_cfg": _ROBOT_29DOF},
    )
    # Domain randomization. The startup-mode terms run once per env at
    # construction and bake their values in for that env's lifetime — with
    # 4096 envs the policy still sees a wide spread of physics within each
    # iter's rollout buffer. Per-reset mass/gain randomization isn't
    # well-supported on implicit/DC actuators (CPU-tensor write), so we
    # stick to startup for those.
    randomize_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    # ±15% body mass perturbation — covers payload variation (battery,
    # cabling, hand attachments) and minor wear on real hardware.
    randomize_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    # ±30% PD gain perturbation — emulates real motor stiffness/damping
    # ripple, friction, and torque deadband that the implicit-actuator
    # sim treats as perfect.
    randomize_actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": _ROBOT_29DOF,
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # Center-of-mass shift per body — covers battery/cable placement
    # variance, mounted attachments, and inertia model errors. ±3cm is
    # roughly the deviation between Unitree's spec'd CoM and a real
    # G1 with a payload swap.
    randomize_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.01, 0.01)},
        },
    )
    # Random pushes during episodes — proxy for unmodelled disturbances
    # (foot scuffing, contact glitches, slight ground non-flatness).
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
    )


# -----------------------------------------------------------------------------
# Full env cfg
# -----------------------------------------------------------------------------

@configclass
class G1MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
    # Motion clip. Sentinel strings select built-in synthetic motions
    # (see envs.g1_tracking_env._load_motion); otherwise this is an NPZ path.
    motion_path: str = "synthetic"
    motion_fps_override: float | None = None

    # Action latency for sim-to-real robustness. Each env's action is held in
    # a FIFO buffer and applied N steps later, where N is sampled uniformly
    # from [0, action_latency_max_steps] at startup (re-sampled on reset).
    # At 50 Hz control, 0 → no delay; 1 → 20 ms; 2 → 40 ms. Real G1 actuator
    # round-trip is ~5-20 ms, so max_steps=2 covers it. 0 disables the
    # mechanism (zero overhead, identical to the pre-latency env).
    action_latency_max_steps: int = 0

    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Whole-body PD control loop runs at 50 Hz (dt 5 ms x decim 4).
    decimation: int = 4
    episode_length_s: float = 10.0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0)
