import argparse

import examples
import numpy as np
from mpmath import mp

import eig3x3

mp.dps = 256
np.set_printoptions(precision=32)

parser = argparse.ArgumentParser()
parser.add_argument("--D", type=str, default="d2")
parser.add_argument("--U", type=str, default="u1")
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=1.0,
                    help="Conditioning deteriorating scale for the U matrix")

args = parser.parse_args()

num_points = 200
deltas = np.logspace(-16, 0, num_points)
variants = ["c", "naive", "lapack", "naive_tensor"]
metrics = ["J2", "J3", "disc", "eig0", "eig1", "eig2"]


def dev(A):
    """Return the deviatoric part of a matrix A."""
    trace = A[0, 0] + A[1, 1] + A[2, 2]
    return A - trace / 3 * mp.eye(3)


def cof(A):
    """Return the cofactor matrix of A."""
    return mp.det(A) * mp.inverse(A).T


def safe_compute(func, A, variant, default=np.inf):
    """Safely compute a function, returning default on error."""
    try:
        return func(A, variant=variant)
    except NotImplementedError:
        return default


# Initialize storage
data = []
eps = np.finfo(np.float64).eps

U = examples.U(args.U, args.gamma)
condU = np.linalg.cond(U)

print("=" * 79)
print(
    f"Computing invariants for U={args.U}, D={args.D}, scale={args.scale}, gamma={args.gamma}")
print("---")
print(f"Condition number κ(U) = {np.linalg.cond(U):.4g}")
print()
print("Diagonal matrix D(delta = 0):")
_D = examples.D(args.D, 0, args.scale)
print(_D)
print()
print("Eigenvector matrix U:")
print(U)
print()
print("Matrix A = U D U^-1:")
print(U @ _D @ np.linalg.inv(U))
print("=" * 79)

for delta in deltas:
    D = examples.D(args.D, delta, args.scale)
    A = U @ D @ np.linalg.inv(U)

    # Add noise
    rng = np.random.default_rng(42)
    A += rng.random((3, 3)) * 4 * eps * np.linalg.norm(A, ord="fro")
    A_mp = mp.matrix(A)

    # Compute exact values
    J2_exact = safe_compute(eig3x3.J2, A_mp, "naive_tensor")
    J3_exact = safe_compute(eig3x3.J3, A_mp, "naive_tensor")
    disc_exact = safe_compute(eig3x3.disc, A_mp, "naive_tensor")
    eigs_exact = safe_compute(eig3x3.eigvals, A_mp, "naive_tensor")

    # Compute condition numbers
    dJ2 = dev(A_mp).T
    cond_J2 = mp.mnorm(dJ2, p="f") ** 2 * eps

    try:
        dJ3 = dev(cof(dev(A_mp)))
        cond_J3 = mp.mnorm(dJ3, p="f") * mp.mnorm(dev(A_mp), p="f") * eps
        # cond_J3 = mp.mnorm(dJ3, p="f") * mp.mnorm(A_mp, p="f") * eps
    except:
        cond_J3 = np.inf

    ddisc = (12 * J2_exact**2 * dJ2 - 54 * J3_exact * dJ3)
    cond_disc = mp.mnorm(ddisc, p="f") * mp.mnorm(dev(A_mp), p="f") * eps
    # cond_disc = mp.mnorm(ddisc, p="f") * mp.mnorm(A_mp, p="f") * eps

    eig_cond = condU * mp.mnorm(A_mp, p="f") * eps

    row = [delta]

    # Compute for each variant
    for variant in variants:
        J2 = safe_compute(eig3x3.J2, A, variant)
        J3 = safe_compute(eig3x3.J3, A, variant)
        disc = safe_compute(eig3x3.disc, A, variant)

        try:
            eigs = eig3x3.eigvals(A, variant)
        except NotImplementedError:
            eigs = np.zeros(3)

        # Add values and errors
        values = [J2, J3, disc, eigs[0], eigs[1], eigs[2]]
        exact_values = [J2_exact, J3_exact, disc_exact,
                        eigs_exact[0], eigs_exact[1], eigs_exact[2]]

        for val, exact in zip(values, exact_values):
            row.extend([val, abs(val - exact)])

    # Add condition numbers
    conds = [cond_J2, cond_J3, cond_disc, eig_cond, eig_cond, eig_cond]
    row.extend(conds)

    data.append(row)

# Create header
header = ["delta"]
for variant in variants:
    for metric in metrics:
        header.extend([f"{metric}_values_{variant}",
                      f"{metric}_errors_{variant}"])
for metric in metrics:
    header.append(f"{metric}_conds")

# Save results
result_array = np.array(data)

# Convert mpmath objects to float for formatting
data_float = []
for row in data:
    float_row = []
    for val in row:
        if hasattr(val, '__float__'):  # mpmath objects have __float__ method
            float_row.append(float(val))
        else:
            float_row.append(val)
    data_float.append(float_row)

# Calculate column widths for proper padding
col_widths = []
for i, col_header in enumerate(header):
    max_width = len(col_header)
    for row in data_float:
        if i < len(row):
            # Format the number and check its width
            formatted_val = f"{row[i]:.16e}"
            max_width = max(max_width, len(formatted_val))
    col_widths.append(max_width + 2)  # Add padding

# Create formatted output
with open(f"invariants-{args.D}-{args.U}.dat", "w") as f:
    # Write header with proper padding
    header_line = ""
    for i, col_header in enumerate(header):
        header_line += col_header.ljust(col_widths[i])
    f.write(header_line.rstrip() + "\n")

    # Write data rows with proper padding
    for row in data_float:
        data_line = ""
        for i, val in enumerate(row):
            if i < len(col_widths):
                formatted_val = f"{val:.16e}"
                data_line += formatted_val.ljust(col_widths[i])
        f.write(data_line.rstrip() + "\n")
