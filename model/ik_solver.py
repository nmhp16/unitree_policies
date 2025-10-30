import pinocchio as pin
import numpy as np

def solve_ik(model, data, frame_name, target_pos, target_rot=None, q_init=None,
                tol=1e-3, max_iter=100, alpha=0.5):
    """
    Solve IK for given frame to reach target_pos
    """
    frame_id = model.getFrameId(frame_name)
    if q_init is None:
        q = pin.utils.zero(model.nq)
    else:
        q = q_init.copy()

    if target_rot is None:
        target_rot = np.eye(3)
    goal = pin.SE3(target_rot, target_pos)

    for it in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id]
        err = pin.log6(current.inverse() * goal).vector # 6D error vector
        if np.linalg.norm(err) < tol:
            break
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)
        dq = alpha * np.linalg.pinv(J) @ err
        q += dq

    return q, np.linalg.norm(err)