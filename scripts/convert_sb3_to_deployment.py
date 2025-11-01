"""
Convert Stable-Baselines3 model to PyTorch JIT format for deployment
This creates a standalone .pt file that can be used with MuJoCo
"""

import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
import os


class DeploymentPolicy(torch.nn.Module):
    """Wrapper to make SB3 policy compatible with MuJoCo deployment"""
    
    def __init__(self, sb3_model):
        super().__init__()
        # Extract just the actor network
        self.actor = sb3_model.policy.mlp_extractor.policy_net
        self.action_net = sb3_model.policy.action_net
        
    def forward(self, obs):
        """Forward pass for deployment"""
        features = self.actor(obs)
        actions = self.action_net(features)
        return actions


def convert_model(model_path: str, output_path: str = None):
    """
    Convert SB3 model to deployment format
    
    Args:
        model_path: Path to .zip model file
        output_path: Where to save .pt file (optional)
    """
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create deployment policy
    print("Creating deployment policy...")
    deploy_policy = DeploymentPolicy(model)
    deploy_policy.eval()
    
    # Test with dummy input
    print("Testing policy...")
    dummy_obs = torch.randn(1, 47)  # G1 has 47-dim observation
    with torch.no_grad():
        output = deploy_policy(dummy_obs)
    print(f"[OK] Policy works! Output shape: {output.shape}")
    
    # Trace for deployment
    print("Tracing model...")
    traced = torch.jit.trace(deploy_policy, dummy_obs)
    
    # Save
    if output_path is None:
        # Save next to the original model
        output_path = model_path.replace('.zip', '_policy.pt')
        if output_path == model_path:  # In case it wasn't a .zip
            output_path = model_path + '_policy.pt'
    
    traced.save(output_path)
    print(f"[OK] Deployment model saved to: {output_path}")
    
    # Verify the saved model works
    print("Verifying saved model...")
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        test_output = loaded(dummy_obs)
    print(f"[OK] Loaded model works! Output shape: {test_output.shape}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SB3 model to deployment format")
    parser.add_argument("model_path", type=str, help="Path to .zip model file")
    parser.add_argument("--output", type=str, default=None, help="Output .pt file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        exit(1)
    
    print("=" * 60)
    print("SB3 to Deployment Converter")
    print("=" * 60)
    
    output_path = convert_model(args.model_path, args.output)
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nTo use with MuJoCo:")
    print(f"  1. Edit deploy/deploy_mujoco/configs/g1.yaml")
    print(f"  2. Set policy_path: \"{output_path}\"")
    print(f"  3. Run: python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml")
    print()
