"""
Gymnasium-compatible G1 environment using MuJoCo
This allows training on Mac without Isaac Gym
"""

import gymnasium as gym
import numpy as np
import mujoco
import os
from pathlib import Path
from typing import Optional, Tuple


class G1GymnasiumEnv(gym.Env):
    """G1 humanoid robot environment for Gymnasium + MuJoCo"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Load MuJoCo model
        self.model_path = self._get_model_path()
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Simulation parameters (match Isaac Gym config)
        self.dt = 0.002  # 500Hz physics
        self.decimation = 10  # 50Hz control
        self.max_episode_steps = 1000  # 20 seconds
        
        # PD controller gains (from g1_config.py)
        self.kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
        self.kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
        
        # Default joint positions (from g1_config.py)
        self.default_dof_pos = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0   # right leg
        ], dtype=np.float32)
        
        # Observation/action scaling (from g1_config.py)
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
        
        # State tracking
        self.step_count = 0
        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # vx, vy, omega_z
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Initial state
        self.initial_qpos = self._get_initial_qpos()
        
    def _get_model_path(self) -> str:
        """Get path to G1 MuJoCo XML file"""
        current_file = Path(__file__).resolve()
        repo_root = current_file.parents[2]  # Go up to project root
        xml_path = repo_root / "robots" / "g1_description" / "g1_12dof.xml"
        
        if not xml_path.exists():
            raise FileNotFoundError(f"G1 model not found at {xml_path}")
        
        return str(xml_path)
    
    def _get_initial_qpos(self) -> np.ndarray:
        """Initial robot configuration"""
        qpos = np.zeros(self.model.nq)
        # Base position [x, y, z, qw, qx, qy, qz]
        qpos[0:3] = [0.0, 0.0, 0.8]  # xyz position
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion (w, x, y, z)
        # Joint positions
        qpos[7:] = self.default_dof_pos
        return qpos
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # Reset state tracking
        self.step_count = 0
        self.last_action = np.zeros(12, dtype=np.float32)
        
        # Randomize command (optional)
        if options and options.get('randomize_command', True):
            self.command[0] = np.random.uniform(0.0, 1.0)  # forward vel
            self.command[1] = np.random.uniform(-0.5, 0.5)  # lateral vel
            self.command[2] = np.random.uniform(-0.5, 0.5)  # turn rate
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one control step"""
        # Clip and store action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.last_action = action
        
        # Convert action to target joint positions
        target_dof_pos = action * self.action_scale + self.default_dof_pos
        
        # Run physics for decimation steps (50Hz control, 500Hz physics)
        for _ in range(self.decimation):
            # PD control
            tau = self._compute_torques(target_dof_pos)
            self.data.ctrl[:] = tau
            
            # Step physics
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Get observation, reward, termination
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'step': self.step_count,
            'base_height': self.data.qpos[2],
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_torques(self, target_dof_pos: np.ndarray) -> np.ndarray:
        """PD controller to compute joint torques"""
        current_pos = self.data.qpos[7:]  # Joint positions
        current_vel = self.data.qvel[6:]  # Joint velocities
        
        tau = (target_dof_pos - current_pos) * self.kps + (0.0 - current_vel) * self.kds
        return tau
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct 47-dim observation vector (matches Isaac Gym version):
        - ang_vel (3)
        - projected_gravity (3)
        - commands (3)
        - dof_pos (12)
        - dof_vel (12)
        - actions (12)
        - phase (2: sin, cos)
        """
        # Base orientation and angular velocity
        quat = self.data.qpos[3:7]  # [qw, qx, qy, qz]
        ang_vel = self.data.qvel[3:6] * self.ang_vel_scale
        
        # Projected gravity (gravity vector in base frame)
        gravity = self._get_gravity_orientation(quat)
        
        # Joint positions and velocities
        dof_pos = (self.data.qpos[7:] - self.default_dof_pos) * self.dof_pos_scale
        dof_vel = self.data.qvel[6:] * self.dof_vel_scale
        
        # Gait phase (for rhythmic locomotion)
        period = 0.8  # seconds
        phase = (self.step_count * self.dt * self.decimation) % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        
        # Construct observation
        obs = np.concatenate([
            ang_vel,                      # 3
            gravity,                      # 3
            self.command * self.cmd_scale,  # 3
            dof_pos,                      # 12
            dof_vel,                      # 12
            self.last_action,             # 12
            [sin_phase, cos_phase]        # 2
        ]).astype(np.float32)
        
        return obs
    
    def _get_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        """Compute gravity vector in base frame from quaternion"""
        qw, qx, qy, qz = quat
        
        # Rotation matrix to transform world gravity to base frame
        # Gravity in world frame is [0, 0, -1]
        gx = 2 * (-qz * qx + qw * qy)
        gy = -2 * (qz * qy + qw * qx)
        gz = 1 - 2 * (qw * qw + qz * qz)
        
        return np.array([gx, gy, gz], dtype=np.float32)
    
    def _compute_reward(self) -> float:
        """
        Compute reward (ported from g1_env.py reward scales)
        """
        reward = 0.0
        
        # Linear velocity tracking
        base_lin_vel = self.data.qvel[0:3]
        lin_vel_error = np.sum(np.square(self.command[:2] - base_lin_vel[:2]))
        reward += np.exp(-lin_vel_error / 0.25) * 1.0  # tracking_lin_vel
        
        # Angular velocity tracking
        base_ang_vel = self.data.qvel[3:6]
        ang_vel_error = np.square(self.command[2] - base_ang_vel[2])
        reward += np.exp(-ang_vel_error / 0.25) * 0.5  # tracking_ang_vel
        
        # Penalize xy angular velocity
        reward -= np.sum(np.square(base_ang_vel[:2])) * 0.05
        
        # Penalize z linear velocity
        reward -= np.square(base_lin_vel[2]) * 2.0
        
        # Orientation penalty (stay upright)
        quat = self.data.qpos[3:7]
        gravity = self._get_gravity_orientation(quat)
        reward -= np.square(gravity[2] - 1.0) * 1.0
        
        # Base height penalty
        base_height = self.data.qpos[2]
        target_height = 0.78
        reward -= np.square(base_height - target_height) * 10.0
        
        # Joint velocity penalty
        dof_vel = self.data.qvel[6:]
        reward -= np.sum(np.square(dof_vel)) * 1e-3
        
        # Action rate penalty (smoothness)
        # This would require storing previous action
        
        # Alive bonus
        reward += 0.15
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if robot falls
        base_height = self.data.qpos[2]
        if base_height < 0.3:  # Too low
            return True
        
        # Terminate if robot tilts too much
        quat = self.data.qpos[3:7]
        gravity = self._get_gravity_orientation(quat)
        if gravity[2] < 0.3:  # Tilted more than ~70 degrees
            return True
        
        return False
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # TODO: Implement offscreen rendering if needed
            pass
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Register environment with Gymnasium
gym.register(
    id='G1-v0',
    entry_point='legged_gym.envs.g1.g1_gymnasium_env:G1GymnasiumEnv',
    max_episode_steps=1000,
)
