import numpy as np


def U(name, gamma):
    matrices = {
        "ident": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "upper": np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
        "tough": np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1]]),
        "worst": np.array([[2, 1, 2], [1, 2, 2], [2, 2, 3]]),
        "u1": np.array([[1, -1, 1], [1, 1, 1], [-1, -1, 1]]),
        "u2": np.array([[1, 1, 1], [1, 0, 1], [2, 1, 2 + gamma]]),
        "u3": np.array([[2/3, 2/3, 1/3], [-2/3, 1/3, 2/3], [1/3, -2/3, 2/3]]),
        "symm": np.array([[1/np.sqrt(2), -1/2, 1/2],
                          [1/np.sqrt(2), 1/2, -1/2],
                          [0, 1/np.sqrt(2), 1/np.sqrt(2)]])
    }
    return matrices[name]


def D(name, delta, a=1):
    """Returns diagonal matrix for a specified test case.

    Naming convention follows pattern of defining first the multiplicity
    of the eigenvalues (single, double, triple), and then invariants
    that are either approaching zero as delta -> 0, or are exactly zero.

    Examples:
    - "single_lim_J3" is a matrix with single eigenvalue,
        where J3 -> 0 as delta -> 0.
    - "double_lim_J3J2" is a matrix with double eigenvalue,
        where J2, J3 -> 0 as delta -> 0.
    - "triple_J3" is a matrix with triple eigenvalue and J3 = 0.
        In this case the naming is ambiguous, since J2 = 0 and disc = 0 as well.
    """
    matrices = {
        "single": np.diag([-1, 1, 2 + 2 * delta]) * a / 4,
        "single_lim_J3": np.diag([-1 - delta, 0, 1 + 2 * delta]) * a / 4,
        "single_lim_disc_t": np.diag([-1, 1, 1 + delta]) * a,
        "single_lim_disc_n": np.diag([0, 2 - delta, 2 + delta]) * a / 2,
        "single_lim_J3J2": np.diag([1 - delta, 1, 1 + 2 * delta]) * a,
        "single_J3": np.diag([-1 - delta, 0, 1 + delta]) * a / 2,
        "single_J3_lim_J2": np.diag([1 - delta, 1, 1 + delta]) * a,
        "double": np.diag([-1 - delta, 1, 1]) * a,
        "double_lim_J3J2": np.diag([1, 1, 1 + delta]) * a,
        "single_J2_neg": np.diag([1 - 2j, 4 + 4 * delta, 1 + 2j]) * a / 4,
        "single_J2_pos": np.diag([1 - 1j, 1 + 1j, 4 + 4 * delta]) * a / 4,
        "triple_J3": np.diag([- delta, 0, delta]),
        "d3": np.diag([0, 1, 2 + delta]) * a
    }
    return matrices[name]
