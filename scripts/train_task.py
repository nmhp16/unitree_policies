"""
Universal Training Script for Multiple Tasks
Trains any task defined in configs/tasks/
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
import json

# Add envs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'envs', 'g1'))
from g1_gymnasium_env import G1GymnasiumEnv


def load_task_config(task_name: str) -> dict:
    """Load task configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "configs" / "tasks" / f"{task_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}\n"
                                f"Available tasks: {list_available_tasks()}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def list_available_tasks() -> list:
    """List all available task configurations"""
    tasks_dir = Path(__file__).parent.parent / "configs" / "tasks"
    if not tasks_dir.exists():
        return []
    return [f.stem for f in tasks_dir.glob("*.yaml")]


def make_env(rank: int, seed: int = 0):
    """Create environment instance"""
    def _init():
        env = G1GymnasiumEnv()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def get_model_save_path(task_name: str, version: str = None) -> Path:
    """
    Get path for saving model
    Format: models/{task_name}/v{version}_{steps}_{timestamp}/
    """
    models_dir = Path(__file__).parent.parent / "models" / task_name
    
    if version is None:
        # Auto-generate version based on existing models
        existing = list(models_dir.glob("v*")) if models_dir.exists() else []
        versions = [int(d.name.split('_')[0][1:]) for d in existing if d.name.startswith('v')]
        next_version = max(versions) + 1 if versions else 1
        version = str(next_version)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = models_dir / f"v{version}_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    return save_path


def save_training_info(save_path: Path, config: dict, args: argparse.Namespace):
    """Save training configuration and metadata"""
    info = {
        'task': config['task']['name'],
        'description': config['task']['description'],
        'robot': config['robot']['model'],
        'training_config': config['training'],
        'reward_weights': config['rewards'],
        'command_line_args': vars(args),
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
    }
    
    with open(save_path / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Also save the full config
    with open(save_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_git_commit():
    """Get current git commit hash"""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return "unknown"


def train(args):
    """Main training function"""
    
    # Load task configuration
    print(f"\n{'='*60}")
    print(f"Loading task: {args.task}")
    print(f"{'='*60}\n")
    
    config = load_task_config(args.task)
    train_config = config['training']
    
    # Override config with command line args if provided
    n_envs = args.n_envs if args.n_envs else train_config['n_envs']
    total_timesteps = args.total_timesteps if args.total_timesteps else train_config['total_timesteps']
    learning_rate = args.learning_rate if args.learning_rate else train_config['learning_rate']
    
    print(f"Task: {config['task']['name']}")
    print(f"Description: {config['task']['description']}")
    print(f"Robot: {config['robot']['model']}")
    print(f"Number of environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Create save directory
    save_path = get_model_save_path(args.task, args.version)
    print(f"Saving to: {save_path}\n")
    
    # Save training info
    save_training_info(save_path, config, args)
    
    # Create vectorized environments
    print(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i, args.seed) for i in range(n_envs)])
    env = VecMonitor(env, str(save_path / "training"))
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(n_envs, args.seed)])
    eval_env = VecMonitor(eval_env, str(save_path / "eval"))
    
    print("[OK] Environments created\n")
    
    # Configure PPO
    print("Creating PPO agent...")
    policy_config = train_config['policy']
    net_arch = [dict(pi=policy_config['net_arch']['pi'], 
                     vf=policy_config['net_arch']['vf'])]
    
    activation_fn = {
        'relu': torch.nn.ReLU,
        'elu': torch.nn.ELU,
        'tanh': torch.nn.Tanh,
    }[policy_config['activation']]
    
    model = PPO(
        policy_config['type'],
        env,
        learning_rate=learning_rate,
        n_steps=train_config['n_steps'],
        batch_size=train_config['batch_size'],
        n_epochs=train_config['n_epochs'],
        gamma=train_config['gamma'],
        gae_lambda=train_config['gae_lambda'],
        clip_range=train_config['clip_range'],
        ent_coef=train_config['ent_coef'],
        vf_coef=train_config['vf_coef'],
        max_grad_norm=train_config['max_grad_norm'],
        verbose=1,
        tensorboard_log=str(save_path / "tensorboard"),
        device='cpu',
        policy_kwargs={
            'net_arch': net_arch,
            'activation_fn': activation_fn,
        }
    )
    
    print("[OK] PPO agent created\n")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_config['save_freq'],
        save_path=str(save_path / "checkpoints"),
        name_prefix=f"{args.task}_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval"),
        eval_freq=train_config['eval_freq'],
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    # Train
    print("Starting training...\n")
    print(f"Monitor progress:")
    print(f"  - TensorBoard: tensorboard --logdir={save_path / 'tensorboard'}")
    print(f"  - Checkpoints: {save_path / 'checkpoints'}")
    print(f"  - Best model: {save_path / 'best_model'}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    final_model_path = save_path / "final_model"
    model.save(str(final_model_path))
    print(f"\n[OK] Final model saved to: {final_model_path}")
    
    # Save final metrics
    metrics = {
        'total_timesteps': total_timesteps,
        'final_eval_reward': eval_callback.best_mean_reward,
        'training_duration_seconds': eval_callback.last_mean_reward,
    }
    with open(save_path / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create symlink to best if requested
    if args.mark_as_best:
        best_link = save_path.parent / "best"
        if best_link.exists():
            best_link.unlink()
        best_link.symlink_to(save_path.name)
        print(f"[OK] Marked as best model: {best_link}")
    
    env.close()
    eval_env.close()
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"\nModel saved to: {save_path}")
    print(f"\nTo deploy this model:")
    print(f"  1. Update deploy/configs/{args.task}.yaml")
    print(f"  2. Set policy_path to: {final_model_path}.zip")
    print(f"  3. Run: python deploy/deploy_mujoco.py {args.task}.yaml\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train G1 robot for any task")
    
    # Task selection
    parser.add_argument("task", type=str, 
                        help=f"Task to train. Available: {list_available_tasks()}")
    
    # Training overrides
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments (overrides config)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total training timesteps (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    
    # Model versioning
    parser.add_argument("--version", type=str, default=None,
                        help="Model version (auto-increments if not specified)")
    parser.add_argument("--mark-as-best", action="store_true",
                        help="Create 'best' symlink to this model")
    
    # Quick test mode
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test mode (10k timesteps, 2 envs)")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.total_timesteps = 10000
        args.n_envs = 2
        print("\n[Quick Test Mode] - Training for 10k timesteps with 2 envs\n")
    
    train(args)
