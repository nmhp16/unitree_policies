import pinocchio as pin
import os

def load_g1_model(urdf_path="g1_29dof.urdf", package_dirs=None):
    base = os.path.dirname(os.path.abspath(__file__))
    urdf_full = os.path.join(base, urdf_path)

    if not os.path.exists(urdf_full):
        raise FileNotFoundError(f"URDF file not found: {urdf_full}")
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_full, package_dirs or [os.path.dirname(urdf_full)]
    )
    data = model.createData()
    print(f"Loaded model '{model.name}' with {model.nq} joints.")
    return model, data

if __name__ == "__main__":
    g1_model, g1_data = load_g1_model("g1_29dof.urdf")