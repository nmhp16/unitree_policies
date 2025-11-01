"""
Test the G1 Gymnasium environment
Quick verification that everything works before training
"""

import sys
import os
import numpy as np

# Add the parent directory to path to import without triggering legged_gym.__init__
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'envs', 'g1'))
from g1_gymnasium_env import G1GymnasiumEnv


def test_environment():
    """Test basic environment functionality"""
    
    print("=" * 60)
    print("Testing G1 Gymnasium Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    try:
        env = G1GymnasiumEnv()
        print("   [OK] Environment created successfully")
        print(f"   - Observation space: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space.shape}")
    except Exception as e:
        print(f"   [ERROR] Failed to create environment: {e}")
        return False
    
    # Test reset
    print("\n2. Testing reset...")
    try:
        obs, info = env.reset(seed=42)
        print("   [OK] Reset successful")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Observation sample: {obs[:5]}")
    except Exception as e:
        print(f"   [ERROR] Reset failed: {e}")
        return False
    
    # Test multiple steps
    print("\n3. Testing simulation steps...")
    try:
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                print(f"   [WARN] Episode terminated at step {i+1}")
                break
        
        print(f"   [OK] Completed {i+1} steps")
        print(f"   - Total reward: {total_reward:.2f}")
        print(f"   - Base height: {info.get('base_height', 'N/A'):.3f}m")
    except Exception as e:
        print(f"   [ERROR] Step failed: {e}")
        return False
    
    # Test full episode
    print("\n4. Running full episode...")
    try:
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        for steps in range(env.max_episode_steps):
            # Simple policy: zero action (try to stand still)
            action = np.zeros(12, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"   [OK] Episode completed")
        print(f"   - Steps: {steps+1}/{env.max_episode_steps}")
        print(f"   - Total reward: {episode_reward:.2f}")
        print(f"   - Average reward: {episode_reward/(steps+1):.3f}")
        print(f"   - Terminated: {terminated}, Truncated: {truncated}")
    except Exception as e:
        print(f"   [ERROR] Episode failed: {e}")
        return False
    
    # Test observation components
    print("\n5. Checking observation structure...")
    try:
        obs, info = env.reset()
        
        ang_vel = obs[0:3]
        gravity = obs[3:6]
        commands = obs[6:9]
        dof_pos = obs[9:21]
        dof_vel = obs[21:33]
        actions = obs[33:45]
        phase = obs[45:47]
        
        print(f"   [OK] Observation components:")
        print(f"   - Angular velocity: {ang_vel}")
        print(f"   - Projected gravity: {gravity}")
        print(f"   - Commands: {commands}")
        print(f"   - Joint positions: shape {dof_pos.shape}")
        print(f"   - Joint velocities: shape {dof_vel.shape}")
        print(f"   - Last actions: shape {actions.shape}")
        print(f"   - Phase (sin, cos): {phase}")
    except Exception as e:
        print(f"   [ERROR] Observation structure check failed: {e}")
        return False
    
    # Clean up
    env.close()
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Train: python train_gymnasium_local.py")
    print("  2. Quick test: python train_gymnasium_local.py --total-timesteps 50000 --n-envs 4")
    print("  3. Full training: python train_gymnasium_local.py --total-timesteps 5000000 --n-envs 8")
    print()
    
    return True


if __name__ == "__main__":
    import sys
    success = test_environment()
    sys.exit(0 if success else 1)
