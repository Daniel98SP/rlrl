import numpy as np


def quat_to_euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    sinp = 2 * (w * y - z * x)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)

    roll = np.arctan2(sinr_cosp, cosr_cosp)
    if abs(sinp) > 1:
        pitch = np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([-pitch, yaw, -roll])


# From RLUtilities
def quat_to_rot_mtx(quat: np.ndarray) -> np.ndarray:
    w = -quat[0]
    x = -quat[1]
    y = -quat[2]
    z = -quat[3]

    theta = np.zeros((3, 3))

    norm = np.dot(quat, quat)
    if norm != 0:
        s = 1.0 / norm

        # front direction
        theta[0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[1, 0] = 2.0 * s * (x * y + z * w)
        theta[2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[0, 1] = 2.0 * s * (x * y - z * w)
        theta[1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[0, 2] = 2.0 * s * (x * z + y * w)
        theta[1, 2] = 2.0 * s * (y * z - x * w)
        theta[2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return theta
