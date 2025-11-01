"""
Train G1 robot using Gymnasium + MuJoCo + Stable-Baselines3
Portable training script that works on any platform
"""

import argparse
import os
import sys
from datetime import datetime
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

# Add path to import custom environment
sys.path.insert(0, str(Path(__file__).parent.parent))
from g1_gymnasium_env import G1GymnasiumEnv


def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    
    :param rank: index of the subprocess
    :param seed: the inital seed for RNG
    """
    def _init():
        env = G1GymnasiumEnv()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train(args):
    """Main training function"""
    
    print("=" * 60)
    print("G1 Gymnasium Training (Mac-Native)")
    print("=" * 60)
    print(f"Number of environments: {args.n_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Create log directory
    log_dir = os.path.join(
        "logs", 
        "g1_gymnasium",
        datetime.now().strftime("%b%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}\n")
    
    # Create vectorized environments (parallel training)
    print(f"Creating {args.n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    env = VecMonitor(env, log_dir)
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(args.n_envs, args.seed)])
    eval_env = VecMonitor(eval_env, os.path.join(log_dir, "eval"))
    
    print("[OK] Environments created\n")
    
    # Configure PPO (parameters from legged_gym config)
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=log_dir,
        device='cpu',  # Change to 'mps' for Apple Silicon GPU, or 'cuda' if available
        policy_kwargs={
            'net_arch': [dict(pi=[32, 32], vf=[32, 32])],  # Match g1_config.py
            'activation_fn': torch.nn.ELU,
        }
    )
    
    print("[OK] PPO agent created\n")
    
    # Callbacks for checkpointing and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="g1_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    # Train
    print("Starting training...\n")
    print("Monitor progress:")
    print(f"  - TensorBoard: tensorboard --logdir={log_dir}")
    print(f"  - Checkpoints: {log_dir}/checkpoints/")
    print(f"  - Best model: {log_dir}/best_model/\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False,  # Disable progress bar to avoid tqdm issues
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"\n[OK] Final model saved to: {final_model_path}")
    
    # Convert to PyTorch JIT for deployment
    print("\nConverting to deployment format...")
    try:
        policy = model.policy.to('cpu')
        
        # Create a simple wrapper for deployment
        class DeploymentPolicy(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
            
            def forward(self, obs):
                with torch.no_grad():
                    actions, _ = self.policy(obs, deterministic=True)
                return actions
        
        deploy_policy = DeploymentPolicy(policy)
        traced = torch.jit.script(deploy_policy)
        
        export_path = os.path.join(log_dir, "policy_1.pt")
        traced.save(export_path)
        print(f"[OK] Deployment policy saved to: {export_path}")
        print(f"\nTo test in MuJoCo:")
        print(f"  1. Edit deploy/deploy_mujoco/configs/g1.yaml")
        print(f"  2. Set policy_path to: {export_path}")
        print(f"  3. Run: python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml")
    except Exception as e:
        print(f"[WARN] Could not export deployment format: {e}")
        print("You can still use the .zip model with Stable-Baselines3")
    
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train G1 with Gymnasium + MuJoCo")
    
    # Environment
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
                        help="Total training timesteps (default: 5M, ~12-24 hours on Mac M1)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    
    # PPO hyperparameters (from g1_config.py)
    parser.add_argument("--n-steps", type=int, default=24,
                        help="Number of steps per env per update (default: 24)")
    parser.add_argument("--batch-size", type=int, default=192,
                        help="Minibatch size (default: 192 = 24 * 8)")
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="Number of epochs per update (default: 5)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range (default: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--vf-coef", type=float, default=1.0,
                        help="Value function coefficient (default: 1.0)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm (default: 1.0)")
    
    # Checkpointing
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save checkpoint every N steps (default: 50k)")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Evaluate every N steps (default: 25k)")
    
    args = parser.parse_args()
    
    # Adjust batch size based on n_envs
    if args.batch_size == 192:
        args.batch_size = args.n_steps * args.n_envs
        print(f"Auto-adjusted batch_size to {args.batch_size} (n_steps * n_envs)")
    
    train(args)
