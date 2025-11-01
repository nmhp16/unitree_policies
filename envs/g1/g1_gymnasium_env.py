"""
Gymnasium-compatible G1 environment using MuJoCo
Portable training environment that works on any platform
"""

import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
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
        
        # Simulation parameters (based on legged_gym config)
        self.dt = 0.005  # 200Hz physics (from sim.dt in config)
        self.decimation = 4  # 50Hz control (from control.decimation)
        self.max_episode_steps = 1000  # 20 seconds (episode_length_s=20)
        
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
        self.prev_action = np.zeros(12, dtype=np.float32)  # For action rate penalty
        self.command = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # vx, vy, omega_z
        
        # Foot contact tracking
        self.last_contacts = np.zeros(2, dtype=bool)  # left, right foot
        self.feet_air_time = np.zeros(2, dtype=np.float32)
        
        # Domain randomization (matching domain_rand config)
        self.randomize_friction = True
        self.friction_range = [0.5, 1.25]
        self.randomize_mass = False
        self.mass_range = [-1.0, 1.0]
        self.push_robots = True
        self.push_interval_s = 15
        self.max_push_vel_xy = 1.0
        self.push_timer = 0.0
        
        # Observation noise (matching noise config)
        self.add_noise = True
        self.noise_level = 1.0
        self.noise_scales = {
            'dof_pos': 0.01,
            'dof_vel': 1.5,
            'ang_vel': 0.2,
            'gravity': 0.05
        }
        
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
        xml_path = repo_root / "robots" / "g1_description" / "g1_description" / "g1_12dof.xml"
        
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
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.last_contacts = np.zeros(2, dtype=bool)
        self.feet_air_time = np.zeros(2, dtype=np.float32)
        self.push_timer = 0.0
        
        # Apply domain randomization
        self._randomize_environment()
        
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
        self.prev_action = self.last_action.copy()  # Store for action rate penalty
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
        
        # Apply random pushes for robustness (domain randomization)
        self._apply_push()
        
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
        Construct 47-dim observation vector:
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
        
        # Add observation noise (matching noise config)
        if self.add_noise:
            noise = np.zeros_like(obs)
            # ang_vel noise (indices 0:3)
            noise[0:3] = (np.random.rand(3) * 2 - 1) * self.noise_scales['ang_vel'] * self.noise_level * self.ang_vel_scale
            # gravity noise (indices 3:6)
            noise[3:6] = (np.random.rand(3) * 2 - 1) * self.noise_scales['gravity'] * self.noise_level
            # commands have no noise (indices 6:9)
            # dof_pos noise (indices 9:21)
            noise[9:21] = (np.random.rand(12) * 2 - 1) * self.noise_scales['dof_pos'] * self.noise_level * self.dof_pos_scale
            # dof_vel noise (indices 21:33)
            noise[21:33] = (np.random.rand(12) * 2 - 1) * self.noise_scales['dof_vel'] * self.noise_level * self.dof_vel_scale
            # actions and phase have no noise (indices 33:47)
            
            obs += noise
        
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
        Compute reward matching original LeggedRobot reward structure
        Based on legged_robot.py reward functions
        """
        reward = 0.0
        
        # Get state variables
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        quat = self.data.qpos[3:7]
        gravity = self._get_gravity_orientation(quat)
        base_height = self.data.qpos[2]
        dof_pos = self.data.qpos[7:]
        dof_vel = self.data.qvel[6:]
        
        # Get foot contacts
        left_foot_contact = self._check_foot_contact('left_ankle_roll_link')
        right_foot_contact = self._check_foot_contact('right_ankle_roll_link')
        contacts = np.array([left_foot_contact, right_foot_contact])
        
        # Update feet air time (before using in rewards)
        dt = self.dt * self.decimation
        contact_filt = np.logical_or(contacts, self.last_contacts)
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += dt
        self.feet_air_time[~contact_filt] = 0.0
        
        # --- REWARD COMPONENTS (matching legged_robot.py scales) ---
        
        # tracking_lin_vel (1.0) - track x,y velocity commands
        lin_vel_error = np.sum(np.square(self.command[:2] - base_lin_vel[:2]))
        reward += np.exp(-lin_vel_error / 0.25) * 1.0
        
        # tracking_ang_vel (0.5) - track yaw rate command
        ang_vel_error = np.square(self.command[2] - base_ang_vel[2])
        reward += np.exp(-ang_vel_error / 0.25) * 0.5
        
        # lin_vel_z (-2.0) - penalize vertical movement
        reward += np.square(base_lin_vel[2]) * (-2.0)
        
        # ang_vel_xy (-0.05) - penalize roll/pitch rotation
        reward += np.sum(np.square(base_ang_vel[:2])) * (-0.05)
        
        # orientation (0.0 in config, but often used) - stay upright
        # Disabled in original config, but keeping with 0 weight for now
        # reward += np.square(gravity[2] - 1.0) * (0.0)
        
        # base_height (0.0 in config) - maintain target height
        # Disabled in original, but keeping code for reference
        # target_height = 1.0  # from cfg.rewards.base_height_target
        # reward += np.square(base_height - target_height) * (0.0)
        
        # dof_vel (0.0 in config) - penalize joint velocities
        # reward += np.sum(np.square(dof_vel)) * (0.0)
        
        # dof_acc (-2.5e-7) - penalize joint accelerations
        # Requires storing last_dof_vel - implementing this
        if hasattr(self, 'last_dof_vel'):
            dof_acc = (dof_vel - self.last_dof_vel) / dt
            reward += np.sum(np.square(dof_acc)) * (-2.5e-7)
        
        # action_rate (-0.01) - penalize rapid action changes
        reward += np.sum(np.square(self.last_action - self.prev_action)) * (-0.01)
        
        # collision (-1.0) - penalize non-foot contacts
        # Check if non-foot bodies have contact forces
        collision_penalty = self._check_collision()
        reward += collision_penalty * (-1.0)
        
        # feet_air_time (1.0) - reward long steps
        # Only reward on first contact when moving
        cmd_norm = np.linalg.norm(self.command[:2])
        if cmd_norm > 0.1:  # Only when robot should be moving
            air_time_reward = np.sum((self.feet_air_time - 0.5) * first_contact)
            reward += air_time_reward * 1.0
        
        # torques (-0.00001) - penalize high torques
        # MuJoCo stores actuator forces in data.actuator_force
        torques = self.data.actuator_force[:12]  # First 12 actuators
        reward += np.sum(np.square(torques)) * (-0.00001)
        
        # stand_still (0.0 in config) - penalize motion when commanded to stand
        # Disabled, but keeping for reference
        # if cmd_norm < 0.1:
        #     reward += np.sum(np.abs(dof_pos - self.default_dof_pos)) * (0.0)
        
        # Store last values for next step
        self.last_contacts = contacts
        if not hasattr(self, 'last_dof_vel'):
            self.last_dof_vel = dof_vel.copy()
        else:
            self.last_dof_vel[:] = dof_vel
        
        return float(reward)
    
    def _check_foot_contact(self, body_name: str) -> bool:
        """Check if a specific body is in contact with ground"""
        try:
            body_id = self.model.body(body_name).id
        except KeyError:
            return False
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Get geom bodies
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            
            if geom1_body == body_id or geom2_body == body_id:
                # Check if contact force is significant
                if contact.dist < 0.01:  # Close proximity
                    return True
        return False
    
    def _check_collision(self) -> float:
        """
        Check for collisions on non-foot bodies
        Returns 1.0 if collision detected, 0.0 otherwise
        """
        # Bodies that should NOT make contact (similar to penalize_contacts_on)
        # For G1, typically torso, thighs, shanks (not feet/ankles)
        penalized_bodies = [
            'pelvis', 'torso_link',
            'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
            'left_knee_link',  # thigh/shank
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 
            'right_knee_link'
        ]
        
        collision_count = 0
        for body_name in penalized_bodies:
            try:
                body_id = self.model.body(body_name).id
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    geom1_body = self.model.geom_bodyid[contact.geom1]
                    geom2_body = self.model.geom_bodyid[contact.geom2]
                    
                    if (geom1_body == body_id or geom2_body == body_id):
                        # Check if force is significant (>0.1N)
                        contact_force = np.linalg.norm(self._get_contact_force(i))
                        if contact_force > 0.1:
                            collision_count += 1
                            break
            except (KeyError, AttributeError):
                continue
        
        return float(collision_count > 0)
    
    def _get_contact_force(self, contact_id: int) -> np.ndarray:
        """Get contact force for a specific contact"""
        # MuJoCo contact forces are in contact.frame
        contact = self.data.contact[contact_id]
        c_array = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, contact_id, c_array)
        return c_array[:3]  # Return only normal + friction forces
    
    def _check_termination(self) -> bool:
        """
        Check if episode should terminate
        Based on check_termination() in legged_robot.py:
        - Contact forces on termination_contact_indices > 1.0
        - Roll > 0.8 rad (~46 deg) or Pitch > 1.0 rad (~57 deg)
        """
        # Check for unwanted contacts (collision with non-foot bodies)
        collision = self._check_collision()
        if collision > 0.5:  # If any collision detected
            return True
        
        # Check roll and pitch angles
        quat = self.data.qpos[3:7]  # [qw, qx, qy, qz]
        roll, pitch, yaw = self._quat_to_euler(quat)
        
        if np.abs(pitch) > 1.0 or np.abs(roll) > 0.8:
            return True
        
        return False
    
    def _quat_to_euler(self, quat: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw]"""
        qw, qx, qy, qz = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def _randomize_environment(self):
        """
        Apply domain randomization (matching domain_rand config)
        - Randomize friction: range [0.5, 1.25]
        - Randomize base mass: range [-1, 1] kg (disabled by default)
        """
        if self.randomize_friction:
            # Randomize ground plane friction
            friction = np.random.uniform(self.friction_range[0], self.friction_range[1])
            # Apply to all geoms (simplified per-environment randomization)
            for i in range(self.model.ngeom):
                self.model.geom_friction[i, 0] = friction
        
        if self.randomize_mass:
            # Randomize base body mass
            base_body_id = 0  # Assuming first body is the base
            mass_change = np.random.uniform(self.mass_range[0], self.mass_range[1])
            self.model.body_mass[base_body_id] += mass_change
    
    def _apply_push(self):
        """
        Apply random push to robot (matching push_robots domain randomization)
        Emulates external disturbance by adding velocity
        """
        self.push_timer += self.dt * self.decimation
        
        if self.push_robots and self.push_timer >= self.push_interval_s:
            # Random push in x-y plane
            push_vel_x = np.random.uniform(-self.max_push_vel_xy, self.max_push_vel_xy)
            push_vel_y = np.random.uniform(-self.max_push_vel_xy, self.max_push_vel_xy)
            
            # Apply to base linear velocity
            self.data.qvel[0] += push_vel_x
            self.data.qvel[1] += push_vel_y
            
            # Reset timer
            self.push_timer = 0.0
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                # Create passive viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Sync viewer with current state
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
