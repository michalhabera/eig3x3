import examples
import numpy as np
import pytest

from eig3x3 import eigvals, eigvalss

# Base tolerance based on machine epsilon
BASE_TOLERANCE = 10 * np.finfo(np.float64).eps


@pytest.mark.parametrize("u_name", ["u1"])
@pytest.mark.parametrize(
    "d_name",
    [
        "single",
        "single_lim_J3",
        "single_lim_disc_t",
        "single_lim_disc_n",
        "single_lim_J3J2",
        "single_J3",
        "single_J3_lim_J2",
        "double",
        "double_lim_J3J2",
        "triple_J3",
    ],
)
@pytest.mark.parametrize(
    "delta", [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 5.0, 500.0]
)
def test_eigvals_c_accuracy(u_name, d_name, delta):
    """Test eigvals with variant='c' matches LAPACK and exact eigenvalues."""
    gamma = 0.1 if u_name == "u2" else 0
    U = examples.U(u_name, gamma)
    D = examples.D(d_name, delta=delta, a=1.0)
    A = U @ D @ np.linalg.inv(U)

    # Compute condition number and scale tolerance
    cond_U = np.linalg.cond(U, 2)
    norm_A = np.linalg.norm(A, 2)
    tolerance = cond_U * norm_A * BASE_TOLERANCE

    eig_c = np.sort(np.real(eigvals(A, variant="c")))
    eig_lapack = eigvals(A, variant="lapack")
    eig_exact = np.sort(np.diag(D))

    max_error_lapack = np.max(np.abs(eig_c - eig_lapack))
    max_error_exact = np.max(np.abs(eig_c - eig_exact))

    assert max_error_lapack < tolerance, (
        f"Max error vs LAPACK {max_error_lapack:.3e} exceeds "
        f"tolerance {tolerance:.3e} (cond={cond_U:.3e}, norm={norm_A:.3e})"
    )
    assert max_error_exact < tolerance, (
        f"Max error vs exact {max_error_exact:.3e} exceeds "
        f"tolerance {tolerance:.3e} (cond={cond_U:.3e}, norm={norm_A:.3e})"
    )


@pytest.mark.parametrize(
    "d_name",
    [
        "single",
        "single_lim_J3",
        "single_lim_disc_t",
        "single_lim_disc_n",
        "single_lim_J3J2",
        "single_J3",
        "single_J3_lim_J2",
        "double",
        "double_lim_J3J2",
        "triple_J3",
        "d3",
    ],
)
@pytest.mark.parametrize(
    "delta", [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 5.0, 500.0]
)
def test_eigvalss_c_accuracy(d_name, delta):
    """Test eigvalss with variant='c' matches LAPACK and exact eigenvalues."""
    U = examples.U("symm", 0)
    D = examples.D(d_name, delta=delta, a=1.0)
    A = U @ D @ U.T

    # Compute condition number and scale tolerance
    cond_U = np.linalg.cond(U, 2)
    norm_A = np.linalg.norm(A, 2)
    tolerance = cond_U * norm_A * BASE_TOLERANCE

    eig_c = np.sort(eigvalss(A, variant="c"))
    eig_lapack = eigvalss(A, variant="lapack")
    eig_exact = np.sort(np.diag(D))

    max_error_lapack = np.max(np.abs(eig_c - eig_lapack))
    max_error_exact = np.max(np.abs(eig_c - eig_exact))

    assert max_error_lapack < tolerance, (
        f"Max error vs LAPACK {max_error_lapack:.3e} exceeds "
        f"tolerance {tolerance:.3e} (cond={cond_U:.3e}, norm={norm_A:.3e})"
    )
    assert max_error_exact < tolerance, (
        f"Max error vs exact {max_error_exact:.3e} exceeds "
        f"tolerance {tolerance:.3e} (cond={cond_U:.3e}, norm={norm_A:.3e})"
    )


def test_eigvalss_identity():
    """Test eigvalss on identity matrix."""
    A = np.eye(3)

    # Identity has condition number 1 and norm 1
    tolerance = BASE_TOLERANCE

    eig_c = np.sort(eigvalss(A, variant="c"))
    eig_lapack = eigvalss(A, variant="lapack")

    max_error = np.max(np.abs(eig_c - eig_lapack))
    assert max_error < tolerance, (
        f"Max error {max_error:.3e} exceeds tolerance {tolerance:.3e}"
    )
    assert np.allclose(eig_c, [1.0, 1.0, 1.0], atol=tolerance)
