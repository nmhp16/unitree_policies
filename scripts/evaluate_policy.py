"""
Evaluate a trained policy in MuJoCo with visualization
Works directly with Stable-Baselines3 models
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

# Add envs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'envs', 'g1'))
from g1_gymnasium_env import G1GymnasiumEnv


def evaluate_policy(model_path: str, n_episodes: int = 5, render: bool = True, 
                    deterministic: bool = True, command: list = None, 
                    slow_motion: float = 1.0, keep_viewer_open: bool = False):
    """
    Evaluate a trained policy
    
    Args:
        model_path: Path to .zip model file
        n_episodes: Number of episodes to run
        render: Show MuJoCo visualization
        deterministic: Use deterministic actions (no exploration)
        command: Custom command [vx, vy, omega_z] or None for default
        slow_motion: Slow down factor (2.0 = half speed, 0.5 = 2x speed)
        keep_viewer_open: Keep viewer open at end
    """
    
    print("\n" + "="*60)
    print("Policy Evaluation")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print(f"Deterministic: {deterministic}")
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    print("[OK] Model loaded")
    
    # Create environment
    render_mode = "human" if render else None
    env = G1GymnasiumEnv(render_mode=render_mode)
    
    # Override command if specified
    if command:
        env.command = np.array(command, dtype=np.float32)
        print(f"Command: vx={command[0]:.2f} m/s, vy={command[1]:.2f} m/s, omega={command[2]:.2f} rad/s")
    else:
        print(f"Command: vx={env.command[0]:.2f} m/s, vy={env.command[1]:.2f} m/s, omega={env.command[2]:.2f} rad/s")
    
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60 + "\n")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"Episode {episode + 1}/{n_episodes}:")
        
        while not done and steps < env.max_episode_steps:
            # Get action from policy
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Render
            if render:
                env.render()
                # Slow motion
                if slow_motion > 1.0:
                    import time
                    time.sleep((slow_motion - 1.0) * 0.02)  # 0.02s per step at normal speed
            
            done = terminated or truncated
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}: reward={episode_reward:.2f}, height={info['base_height']:.3f}m")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Episode summary
        reason = "terminated" if terminated else ("truncated" if truncated else "completed")
        print(f"  Finished: {steps} steps, total reward={episode_reward:.2f} ({reason})")
        print()
    
    # Keep viewer open if requested
    if keep_viewer_open and render:
        print("\n" + "="*60)
        print("Viewer is open. Press Ctrl+C to close...")
        print("="*60)
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nClosing viewer...")
    
    env.close()
    
    # Final statistics
    print("="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Success rate: {sum(1 for l in episode_lengths if l >= env.max_episode_steps * 0.8) / n_episodes * 100:.1f}%")
    print("="*60 + "\n")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
    }


def test_different_commands(model_path: str):
    """Test policy with different velocity commands"""
    
    print("\n" + "="*60)
    print("Testing Different Commands")
    print("="*60 + "\n")
    
    commands = [
        ([0.3, 0.0, 0.0], "Slow forward"),
        ([0.8, 0.0, 0.0], "Fast forward"),
        ([0.5, 0.2, 0.0], "Forward + right"),
        ([0.5, -0.2, 0.0], "Forward + left"),
        ([0.5, 0.0, 0.3], "Forward + turn right"),
        ([0.5, 0.0, -0.3], "Forward + turn left"),
        ([0.0, 0.0, 0.0], "Stand still"),
    ]
    
    for cmd, description in commands:
        print(f"\n{description}: vx={cmd[0]}, vy={cmd[1]}, omega={cmd[2]}")
        print("-" * 40)
        
        result = evaluate_policy(
            model_path,
            n_episodes=3,
            render=False,
            deterministic=True,
            command=cmd
        )
        
        print(f"Result: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    
    parser.add_argument("model_path", type=str, 
                        help="Path to model.zip file or 'latest' for most recent")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--no-render", action="store_true",
                        help="Don't show visualization")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--command", type=float, nargs=3, default=None,
                        metavar=("VX", "VY", "OMEGA"),
                        help="Custom command: forward lateral angular (default: from training)")
    parser.add_argument("--test-commands", action="store_true",
                        help="Test multiple different commands")
    parser.add_argument("--slow-motion", type=float, default=1.0,
                        help="Slow motion factor: 2.0=half speed, 5.0=very slow (default: 1.0)")
    parser.add_argument("--keep-open", action="store_true",
                        help="Keep viewer window open after episodes finish")
    
    args = parser.parse_args()
    
    # Handle "latest" keyword
    if args.model_path == "latest":
        # Find latest model
        models_dir = Path(__file__).parent.parent / "models"
        all_models = []
        for task_dir in models_dir.glob("*/v*"):
            model_file = task_dir / "final_model.zip"
            if model_file.exists():
                all_models.append(model_file)
        
        if not all_models:
            print("Error: No models found!")
            sys.exit(1)
        
        args.model_path = str(max(all_models, key=lambda p: p.stat().st_mtime))
        print(f"Using latest model: {args.model_path}\n")
    
    # Verify model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)
    
    # Run evaluation
    if args.test_commands:
        test_different_commands(args.model_path)
    else:
        evaluate_policy(
            args.model_path,
            n_episodes=args.n_episodes,
            render=not args.no_render,
            deterministic=not args.stochastic,
            command=args.command,
            slow_motion=args.slow_motion,
            keep_viewer_open=args.keep_open
        )
