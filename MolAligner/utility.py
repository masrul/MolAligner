import numpy as np
from .decorators import to_numpy


@to_numpy
def get_unit_vector(vec):
    return vec / np.linalg.norm(vec)


@to_numpy
def get_angle_between_vectors(u, v):

    angle = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    angle = np.arccos(angle)
    angle = np.rad2deg(angle)

    return angle
