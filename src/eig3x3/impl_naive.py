import numpy as np


def I1(matrix):
    return matrix[0, 0] + matrix[1, 1] + matrix[2, 2]


def J2(matrix):
    A00, A01, A02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    A10, A11, A12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    A20, A21, A22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    return 1/3 * (A00**2 - A00*A11 - A00*A22 + 3*A01*A10 + 3*A02*A20 + A11**2 - A11*A22 + 3*A12*A21 + A22**2)


def J3(matrix):
    A00, A01, A02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    A10, A11, A12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    A20, A21, A22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    return 1/27 * (2*A00**3 - 3*A00**2*A11 - 3*A00**2*A22 + 9*A00*A01*A10
                   + 9*A00*A02*A20 - 3*A00*A11**2 + 12*A00*A11*A22 - 18*A00*A12*A21
                   - 3*A00*A22**2 + 9*A01*A10*A11 - 18*A01*A10*A22 + 27*A01*A12*A20
                   + 27*A02*A10*A21 - 18*A02*A11*A20 + 9*A02*A20*A22 + 2*A11**3
                   - 3*A11**2*A22 + 9*A11*A12*A21 - 3*A11*A22**2 + 9*A12*A21*A22 + 2*A22**3)


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
    if disc_ < 0:
        disc_ = 0.0
    phi = np.atan2(np.sqrt(27 * disc_), 27*J3_)

    # Compute the three eigenvalues
    if J2_ < 0:
        J2_ = 0.0
    sqrt_J2 = np.sqrt(3*J2_)
    lambda1 = (I1_ + 2 * sqrt_J2 * np.cos((phi + 2 * np.pi * 1) / 3)) / 3
    lambda2 = (I1_ + 2 * sqrt_J2 * np.cos((phi + 2 * np.pi * 2) / 3)) / 3
    lambda3 = (I1_ + 2 * sqrt_J2 * np.cos((phi + 2 * np.pi * 3) / 3)) / 3

    return np.array([lambda1, lambda2, lambda3])
