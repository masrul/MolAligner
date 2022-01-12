import numpy as np

# from decorators import to_numpy


def rotation_around_xaxis(angle):
    return rotation_around_vector([1, 0, 0], angle)


def rotation_around_yaxis(angle):
    return rotation_around_vector([0, 1, 0], angle)


def rotation_around_zaxis(angle):
    return rotation_around_vector([0, 0, 1], angle)


# @to_numpy
def rotation_around_vector(vec, angle):
    vec = vec / np.linalg.norm(vec)
    angle = np.deg2rad(angle)

    # quaternions
    q0 = np.cos(angle / 2.0)
    q1 = vec[0] * np.sin(angle / 2.0)
    q2 = vec[1] * np.sin(angle / 2.0)
    q3 = vec[2] * np.sin(angle / 2.0)

    q0q0 = q0 * q0
    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3
    q1q1 = q1 * q1
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q2 = q2 * q2
    q2q3 = q2 * q3
    q3q3 = q3 * q3

    # to_matrix
    rot_mat = np.zeros((3, 3))

    rot_mat[0, 0] = q0q0 + q1q1 - q2q2 - q3q3
    rot_mat[0, 1] = 2 * (q1q2 - q0q3)
    rot_mat[0, 2] = 2 * (q1q3 + q0q2)

    rot_mat[1, 0] = 2 * (q1q2 + q0q3)
    rot_mat[1, 1] = q0q0 - q1q1 + q2q2 - q3q3
    rot_mat[1, 2] = 2 * (q2q3 - q0q1)

    rot_mat[2, 0] = 2 * (q1q3 - q0q2)
    rot_mat[2, 1] = 2 * (q2q3 + q0q1)
    rot_mat[2, 2] = q0q0 - q1q1 - q2q2 + q3q3

    return rot_mat


# @to_numpy
def align_two_vectors(u, v):

    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    cross = np.cross(u, v)
    dot = np.dot(u, v)

    if abs(1 + dot) < 1e-7:
        return -1.0 * np.eye(3, dtype=float)

    rot_mat = np.zeros((3, 3), dtype=float)
    vx = np.zeros((3, 3), dtype=float)
    identity_mat = np.eye(3, dtype=float)

    vx[0, 1] = -cross[2]
    vx[0, 2] = cross[1]

    vx[1, 0] = cross[2]
    vx[1, 2] = -cross[0]

    vx[2, 0] = -cross[1]
    vx[2, 1] = cross[0]

    factor = 1.0 / (1 + dot)

    rot_mat = identity_mat + vx + np.matmul(vx, vx) * factor

    return rot_mat
