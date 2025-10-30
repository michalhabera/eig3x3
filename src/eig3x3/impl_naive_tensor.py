import mpmath as mp
import numpy as np


def I1(matrix):
    return matrix[0, 0] + matrix[1, 1] + matrix[2, 2]


def J2(matrix):
    if isinstance(matrix, np.ndarray):
        devA = matrix - np.trace(matrix) / 3 * np.eye(3, dtype=matrix.dtype)
        return 1/2 * np.trace(devA @ devA)
    elif isinstance(matrix, mp.matrix):
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        devA = matrix - trace / 3 * mp.eye(3)
        devA2 = devA * devA
        return 1/2 * (devA2[0, 0] + devA2[1, 1] + devA2[2, 2])
    else:
        raise TypeError("Unsupported type for matrix A")


def J3(matrix):
    if isinstance(matrix, np.ndarray):
        devA = matrix - np.trace(matrix) / 3 * np.eye(3, dtype=matrix.dtype)
        return np.linalg.det(devA)
    elif isinstance(matrix, mp.matrix):
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        devA = matrix - trace / 3 * mp.eye(3)
        return mp.det(devA)
    else:
        raise TypeError("Unsupported type for matrix A")


def disc(matrix):
    J2_ = J2(matrix)
    J3_ = J3(matrix)
    return 4*J2_**3 - 27*J3_**2


def eigvals(matrix):
    I1_ = I1(matrix)
    J2_ = J2(matrix)
    J3_ = J3(matrix)
    disc_ = disc(matrix)

    # Compute the triple angle
    if isinstance(matrix, np.ndarray):
        if disc_ < 0:
            disc_ = 0.0
        phi = np.arctan2(np.sqrt(27 * disc_), 27*J3_)
    elif isinstance(matrix, mp.matrix):
        if disc_ < 0:
            disc_ = mp.mpf(0)
        phi = mp.atan2(mp.sqrt(27 * disc_), 27*J3_)
    else:
        raise TypeError("Unsupported type for matrix A")

    # Compute the three eigenvalues
    if isinstance(matrix, np.ndarray):
        sqrt_J2 = np.sqrt(np.abs(3*J2_))
        lambda1 = (I1_ + 2 * sqrt_J2 * np.cos((phi + 2 * np.pi * 1) / 3)) / 3
        lambda2 = (I1_ + 2 * sqrt_J2 * np.cos((phi + 2 * np.pi * 2) / 3)) / 3
        lambda3 = (I1_ + 2 * sqrt_J2 * np.cos((phi + 2 * np.pi * 3) / 3)) / 3
        return np.array([lambda1, lambda2, lambda3])
    elif isinstance(matrix, mp.matrix):
        sqrt_J2 = mp.sqrt(mp.fabs(3*J2_))
        lambda1 = (I1_ + 2 * sqrt_J2 * mp.cos((phi + 2 * mp.pi * 1) / 3)) / 3
        lambda2 = (I1_ + 2 * sqrt_J2 * mp.cos((phi + 2 * mp.pi * 2) / 3)) / 3
        lambda3 = (I1_ + 2 * sqrt_J2 * mp.cos((phi + 2 * mp.pi * 3) / 3)) / 3
        return mp.matrix([lambda1, lambda2, lambda3])
    else:
        raise TypeError("Unsupported type for matrix A")
