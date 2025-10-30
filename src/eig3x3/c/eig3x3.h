#ifndef EIG3X3_H
#define EIG3X3_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Second invariant J2 for general 3x3 matrices.
 * @param A 3x3 matrix
 * @return J2 invariant
 */
static inline double J2(const double A[static restrict 3][3]) {
  const double d0 = A[0][0] - A[1][1];
  const double d1 = A[0][0] - A[2][2];
  const double d2 = A[1][1] - A[2][2];

  const double offdiag =
      A[0][1] * A[1][0] + A[0][2] * A[2][0] + A[1][2] * A[2][1];

  const double diag = (d0 * d0 + d1 * d1 + d2 * d2) / 6.0;

  return offdiag + diag;
}

/**
 * Third invariant J3 for general 3x3 matrices.
 * @param A 3x3 matrix
 * @return J3 invariant
 */
static inline double J3(const double A[static restrict 3][3]) {
  const double d0 = A[0][0] - A[1][1];
  const double d1 = A[0][0] - A[2][2];
  const double d2 = A[1][1] - A[2][2];

  const double t1 = d1 + d2;
  const double t2 = d0 - d2;
  const double t3 = -d0 - d1;

  const double offdiag =
      A[0][1] * A[1][2] * A[2][0] + A[0][2] * A[1][0] * A[2][1];
  const double mixed = (A[0][1] * A[1][0] * t1 + A[0][2] * A[2][0] * t2 +
                        A[1][2] * A[2][1] * t3) /
                       3.0;
  const double diag = (t1 * t2 * t3) / 27.0;

  return offdiag + mixed - diag;
}

/**
 * Helper for discriminant calculation (general matrices).
 * @param A 3x3 matrix
 * @param result Array of 14 derivative terms
 */
static inline void dx(const double A[static restrict 3][3],
                      double result[static restrict 14]) {
  const double d0 = A[0][0] - A[1][1];
  const double d1 = A[0][0] - A[2][2];
  const double d2 = A[1][1] - A[2][2];

  result[0] = A[0][1] * A[1][2] * A[2][0] - A[0][2] * A[1][0] * A[2][1];
  result[1] = -A[0][1] * A[0][2] * d2 + A[0][1] * A[0][1] * A[1][2] -
              A[0][2] * A[0][2] * A[2][1];
  result[2] = A[0][1] * A[2][1] * d1 - A[0][1] * A[0][1] * A[2][0] +
              A[0][2] * A[2][1] * A[2][1];
  result[3] = A[0][2] * A[1][2] * d0 + A[0][1] * A[1][2] * A[1][2] -
              A[0][2] * A[0][2] * A[1][0];
  result[4] = A[0][1] * A[1][2] * d1 - A[0][1] * A[0][2] * A[1][0] +
              A[0][2] * A[1][2] * A[2][1];
  result[5] = A[0][2] * A[2][1] * d0 - A[0][1] * A[0][2] * A[2][0] +
              A[0][1] * A[1][2] * A[2][1];
  result[6] = -A[0][2] * A[1][0] * d2 + A[0][1] * A[1][0] * A[1][2] -
              A[0][2] * A[1][2] * A[2][0];
  result[7] = A[1][2] * d0 * d1 - A[0][2] * A[1][0] * d1 +
              A[0][1] * A[1][0] * A[1][2] - A[1][2] * A[1][2] * A[2][1];
  result[8] = A[1][2] * d0 * d1 - A[0][2] * A[1][0] * d0 +
              A[0][2] * A[1][2] * A[2][0] - A[1][2] * A[1][2] * A[2][1];
  result[9] = A[0][1] * d1 * d2 + A[0][2] * A[2][1] * d2 +
              A[0][1] * A[0][2] * A[2][0] - A[0][1] * A[0][1] * A[1][0];
  result[10] = A[0][1] * d1 * d2 + A[0][2] * A[2][1] * d1 +
               A[0][1] * A[1][2] * A[2][1] - A[0][1] * A[0][1] * A[1][0];
  result[11] = -A[0][2] * d0 * d2 + A[0][1] * A[1][2] * d0 +
               A[0][2] * A[1][2] * A[2][1] - A[0][2] * A[0][2] * A[2][0];
  result[12] = A[0][2] * d0 * d2 + A[0][1] * A[1][2] * d2 -
               A[0][1] * A[0][2] * A[1][0] + A[0][2] * A[0][2] * A[2][0];
  result[13] = d0 * d1 * d2 - A[0][1] * A[1][0] * d0 + A[0][2] * A[2][0] * d1 -
               A[1][2] * A[2][1] * d2;
}

/**
 * Transpose a 3x3 matrix.
 * @param A Input matrix
 * @param AT Output transposed matrix (can alias A)
 */
static inline void transpose_3x3(const double A[static restrict 3][3],
                                 double AT[static restrict 3][3]) {
  AT[0][0] = A[0][0];
  AT[0][1] = A[1][0];
  AT[0][2] = A[2][0];
  AT[1][0] = A[0][1];
  AT[1][1] = A[1][1];
  AT[1][2] = A[2][1];
  AT[2][0] = A[0][2];
  AT[2][1] = A[1][2];
  AT[2][2] = A[2][2];
}

/**
 * Discriminant for general 3x3 matrices. Used to determine eigenvalue
 * multiplicity.
 * @param A 3x3 matrix
 * @return Discriminant value
 */
static inline double disc(const double A[static restrict 3][3]) {
  double u[14], v[14];
  double AT[3][3];

  dx(A, u);
  transpose_3x3(A, AT);
  dx(AT, v);

  static const double weights[14] = {
      [0] = 9.0,  [1] = 6.0,  [2] = 6.0,  [3] = 6.0, [4] = 8.0,
      [5] = 8.0,  [6] = 8.0,  [7] = 2.0,  [8] = 2.0, [9] = 2.0,
      [10] = 2.0, [11] = 2.0, [12] = 2.0, [13] = 1.0};

  double sum = 0.0;
  for (size_t i = 0; i < 14; i++) {
    sum = fma(weights[i] * u[i], v[i], sum);
  }

  return sum;
}

/**
 * Compute all three eigenvalues using cubic formula.
 * @param A 3x3 matrix
 * @param eigenvalues Output array for 3 eigenvalues
 */
static inline void eigvals(const double A[static restrict 3][3],
                           double eigenvalues[static restrict 3]) {
  const double I1 = A[0][0] + A[1][1] + A[2][2];
  const double j2 = J2(A);
  const double j3 = J3(A);
  double discriminant = disc(A);

  const double phi = atan2(sqrt(27.0 * discriminant), 27.0 * j3);
  const double sqrt_3j2 = sqrt(3.0 * j2);

  static const double TWO_PI = 2 * 3.1415926535897932;
  const double amplitude = 2.0 * sqrt_3j2;

  for (size_t k = 0; k < 3; k++) {
    const double angle = (phi + TWO_PI * (double)(k + 1)) / 3.0;
    eigenvalues[k] = fma(amplitude, cos(angle), I1) / 3.0;
  }
}

/**
 * Second invariant J2 for symmetric matrices (optimized).
 * @param A Symmetric 3x3 matrix
 * @return J2 invariant
 */
static inline double J2s(const double A[static restrict 3][3]) {
  const double d0 = A[0][0] - A[1][1];
  const double d1 = A[0][0] - A[2][2];
  const double d2 = A[1][1] - A[2][2];

  const double offdiag =
      A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];

  const double diag = (d0 * d0 + d1 * d1 + d2 * d2) / 6.0;

  return offdiag + diag;
}

/**
 * Third invariant J3 for symmetric matrices (optimized).
 * @param A Symmetric 3x3 matrix
 * @return J3 invariant
 */
static inline double J3s(const double A[static restrict 3][3]) {
  const double d0 = A[0][0] - A[1][1];
  const double d1 = A[0][0] - A[2][2];
  const double d2 = A[1][1] - A[2][2];

  const double t1 = d1 + d2;
  const double t2 = d0 - d2;
  const double t3 = -d0 - d1;

  const double offdiag = 2 * A[0][1] * A[1][2] * A[0][2];
  const double mixed = (A[0][1] * A[0][1] * t1 + A[0][2] * A[0][2] * t2 +
                        A[1][2] * A[1][2] * t3) /
                       3.0;
  const double diag = (t1 * t2 * t3) / 27.0;

  return offdiag + mixed - diag;
}

/**
 * Helper for discriminant calculation (symmetric matrices).
 * @param A Symmetric 3x3 matrix
 * @param result Array of 5 derivative terms
 */
static inline void dxs(const double A[static restrict 3][3],
                       double result[static restrict 5]) {
  const double d0 = A[0][0] - A[1][1];
  const double d1 = A[0][0] - A[2][2];
  const double d2 = A[1][1] - A[2][2];

  const double w = A[0][1];
  const double v = A[0][2];
  const double u = A[1][2];

  const double alpha = d2;
  const double beta = -d1;
  const double gamma_ = d0;

  result[0] = 3.0 * sqrt(3.0) * (v * w * alpha + u * (v * v - w * w));
  result[1] =
      alpha * beta * gamma_ + alpha * u * u + beta * v * v + gamma_ * w * w;
  result[2] = 2.0 * u * beta * gamma_ - v * w * (beta - gamma_) +
              u * (2.0 * u * u - v * v - w * w);
  result[3] = 2.0 * (v * alpha * gamma_ + u * w * (beta - gamma_) +
                     v * (v * v + w * w - 2.0 * u * u));
  result[4] = 2.0 * (w * alpha * beta + u * v * (beta - gamma_) +
                     w * (v * v + w * w - 2.0 * u * u));
}

/**
 * Discriminant for symmetric 3x3 matrices (optimized).
 * @param A Symmetric 3x3 matrix
 * @return Discriminant value
 */
static inline double discs(const double A[static restrict 3][3]) {
  double u[5];

  dxs(A, u);

  double sum = 0.0;
  for (size_t i = 0; i < 5; i++) {
    sum = fma(u[i], u[i], sum);
  }
  return sum;
}

/**
 * Compute all three eigenvalues using cubic formula (symmetric matrices,
 * optimized).
 * @param A Symmetric 3x3 matrix
 * @param eigenvalues Output array for 3 eigenvalues
 */
static inline void eigvalss(const double A[static restrict 3][3],
                            double eigenvalues[static restrict 3]) {
  const double I1 = A[0][0] + A[1][1] + A[2][2];
  const double j2 = J2s(A);
  const double j3 = J3s(A);
  double discriminant = discs(A);

  const double phi = atan2(sqrt(27.0 * discriminant), 27.0 * j3);
  const double sqrt_3j2 = sqrt(3.0 * j2);

  static const double TWO_PI = 2 * 3.1415926535897932;
  const double amplitude = 2.0 * sqrt_3j2;

  for (size_t k = 0; k < 3; k++) {
    const double angle = (phi + TWO_PI * (double)(k + 1)) / 3.0;
    eigenvalues[k] = fma(amplitude, cos(angle), I1) / 3.0;
  }
}

#ifdef __cplusplus
}
#endif

#endif /* EIG3X3_H */