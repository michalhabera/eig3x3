import numpy as np

import eig3x3.impl_naive as impl_naive
import eig3x3.impl_naive_tensor as impl_naive_tensor
from eig3x3 import _eig3x3

ffi = _eig3x3.ffi
clib = _eig3x3.lib

__all__ = ["J2", "J3", "J2s", "J3s", "disc", "discs", "eigvals", "eigvalss"]


def eigvals(matrix, variant="c"):
    if variant == "c":
        return _eigvals_c(matrix)
    elif variant == "naive":
        return impl_naive.eigvals(matrix)
    elif variant == "naive_tensor":
        return impl_naive_tensor.eigvals(matrix)
    elif variant == "lapack":
        return np.sort(np.real(np.linalg.eigvals(matrix)))
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def J2(matrix, variant="c"):
    if variant == "c":
        return _J2_c(matrix)
    elif variant == "naive":
        return impl_naive.J2(matrix)
    elif variant == "naive_tensor":
        return impl_naive_tensor.J2(matrix)
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def J3(matrix, variant="c"):
    if variant == "c":
        return _J3_c(matrix)
    elif variant == "naive":
        return impl_naive.J3(matrix)
    elif variant == "naive_tensor":
        return impl_naive_tensor.J3(matrix)
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def disc(matrix, variant="c"):
    if variant == "c":
        return _disc_c(matrix)
    elif variant == "naive":
        return impl_naive.disc(matrix)
    elif variant == "naive_tensor":
        return impl_naive_tensor.disc(matrix)
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def J2s(matrix, variant="c"):
    if variant == "c":
        return _J2s_c(matrix)
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def J3s(matrix, variant="c"):
    if variant == "c":
        return _J3s_c(matrix)
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def discs(matrix, variant="c"):
    if variant == "c":
        return _discs_c(matrix)
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def eigvalss(matrix, variant="c"):
    if variant == "c":
        return _eigvalss_c(matrix)
    elif variant == "lapack":
        return np.sort(np.linalg.eigvalsh(matrix))
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")


def _J2_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    return clib.J2(ptr)


def _J3_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    return clib.J3(ptr)


def _disc_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    return clib.disc(ptr)


def _eigvals_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    eigenvalues = np.zeros(3, dtype=np.float64)
    eig_ptr = ffi.cast("double [3]", eigenvalues.ctypes.data)
    clib.eigvals(ptr, eig_ptr)
    return eigenvalues


def _J2s_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    return clib.J2s(ptr)


def _J3s_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    return clib.J3s(ptr)


def _discs_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    return clib.discs(ptr)


def _eigvalss_c(matrix):
    ptr = ffi.cast("double const [3][3]", matrix.ctypes.data)
    eigenvalues = np.zeros(3, dtype=np.float64)
    eig_ptr = ffi.cast("double [3]", eigenvalues.ctypes.data)
    clib.eigvalss(ptr, eig_ptr)
    return eigenvalues
