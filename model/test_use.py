from pin_model_loader import load_g1_model
from ik_solver import solve_ik
import numpy as np

model, data = load_g1_model("g1_29dof.urdf")
target_pos = np.array([0.4, -0.2, 1.1])
target_rot = np.eye(3)  # No rotation
q, err = solve_ik(model, data, "right_wrist_roll_link", target_pos, target_rot)
print(f"IK result: q = {q}, error norm = {err}")