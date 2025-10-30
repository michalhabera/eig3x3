from pathlib import Path

from cffi import FFI

ffibuilder = FFI()

c_dir = Path(__file__).parent / "c"
ffibuilder.cdef(r"""
    double J2(double const A[restrict static 3][3]);
    double J3(double const A[restrict static 3][3]);
    double disc(double const A[restrict static 3][3]);
    void eigvals(double const A[restrict static 3][3], double eigenvalues[restrict static 3]);

    double J2s(double const A[restrict static 3][3]);
    double J3s(double const A[restrict static 3][3]);
    double discs(double const A[restrict static 3][3]);
    void eigvalss(double const A[restrict static 3][3], double eigenvalues[restrict static 3]);
""")

# Build the library
ffibuilder.set_source(
    "eig3x3._eig3x3",
    '#include "eig3x3.h"',
    include_dirs=[str(c_dir)],
    extra_compile_args=['-O3', '-march=native'],
    libraries=['m'],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
