/*
Build (Linux/OpenBLAS):
  gcc -O3 -march=native -std=c11 performance_benchmark.c -I../src/eig3x3/c \
      -llapacke -lopenblas -lm -o performance_benchmark

  ./performance_benchmark general_matrix.txt
  # override:
  ./performance_benchmark general_matrix.txt 2000 2000
*/

#define _POSIX_C_SOURCE 200809L
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/eig3x3/c/eig3x3.h"

static inline unsigned long long ns_now(void) {
  struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
  return (unsigned long long)ts.tv_sec * 1000000000ull +
         (unsigned long long)ts.tv_nsec;
}

static int read_matrix(const char *path, double A[3][3]) {
  FILE *f = fopen(path, "r");
  if (!f) {
    perror(path);
    return -1;
  }
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      if (fscanf(f, "%lf", &A[i][j]) != 1) {
        fclose(f);
        fprintf(stderr, "read %s failed\n", path);
        return -1;
      }
  fclose(f);
  return 0;
}

typedef int (*eig_fn)(const double A[3][3], double out[3]);

static int wrap_eigvals(const double A[3][3], double out[3]) {
  double M[3][3];
  memcpy(M, A, sizeof M);
  eigvals(M, out);
  return 0;
}
static int wrap_dgeev(const double A[3][3], double out[3]) {
  double a[9], wr[3], wi[3];
  size_t k = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      a[k++] = A[i][j];
  LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', 3, a, 3, wr, wi, NULL, 3, NULL, 3);
  return 0;
}

static void preinit_libs(void) {
  double Id[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, o[3], a[9], wr[3], wi[3];
  eigvals(Id, o);
  size_t k = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      a[k++] = Id[i][j];
  LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', 3, a, 3, wr, wi, NULL, 3, NULL, 3);
}

typedef struct {
  double mean_ns, std_ns, min_ns, max_ns;
} stats_t;

#ifndef OUTER_RUNS
#define OUTER_RUNS 100
#endif
#ifndef INNER_EVALS
#define INNER_EVALS 10000
#endif
#ifndef WARM_RUNS
#define WARM_RUNS 100
#endif

static stats_t bench(eig_fn fn, const double A[3][3], size_t outer,
                     size_t inner) {
  stats_t s = {0};
  s.min_ns = INFINITY;
  s.max_ns = -INFINITY;
  double mean = 0.0, m2 = 0.0;
  size_t n = 0;

  for (size_t k = 0; k < outer + WARM_RUNS; ++k) {
    double out[3];
    unsigned long long t0 = ns_now();
    for (size_t i = 0; i < inner; ++i) {
      fn(A, out);
    }
    unsigned long long t1 = ns_now();
    double ns = (double)(t1 - t0) / (double)inner;

    if (k < (size_t)WARM_RUNS)
      continue;

    if (ns < s.min_ns)
      s.min_ns = ns;
    if (ns > s.max_ns)
      s.max_ns = ns;

    ++n;
    double d = ns - mean;
    mean += d / (double)n;
    m2 += d * (ns - mean);
  }
  s.mean_ns = mean;
  s.std_ns = sqrt(m2 / (double)(n ? n : 1));
  return s;
}

static void print_block(const char *title, const char *n1, eig_fn f1,
                        const char *n2, eig_fn f2, const double A[3][3],
                        size_t outer, size_t inner) {
  stats_t s1 = bench(f1, A, outer, inner);
  stats_t s2 = bench(f2, A, outer, inner);
  printf("=== %s ===\n%-12s %14s %14s %14s %14s\n", title, "Impl", "Mean [ns]",
         "Std [ns]", "Min [ns]", "Max [ns]");
  printf("%-12s %14.2f %14.2f %14.2f %14.2f\n", n1, s1.mean_ns, s1.std_ns,
         s1.min_ns, s1.max_ns);
  printf("%-12s %14.2f %14.2f %14.2f %14.2f\n\n", n2, s2.mean_ns, s2.std_ns,
         s2.min_ns, s2.max_ns);
}

int main(int argc, char **argv) {
  const char *fg = (argc > 1 ? argv[1] : "general_matrix.txt");
  size_t outer = (argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : OUTER_RUNS);
  size_t inner = (argc > 3 ? (size_t)strtoull(argv[3], NULL, 10) : INNER_EVALS);

  double Ag[3][3];
  if (read_matrix(fg, Ag))
    return 1;

  preinit_libs();
  double eig_g[3];
  wrap_eigvals(Ag, eig_g);
  printf("Eigenvalues for general matrix: %.15f %.15f %.15f\n\n", eig_g[0],
         eig_g[1], eig_g[2]);
  printf(
      "Runs: %zu, evals/run: %zu (per-eval stats; first %d runs skipped)\n\n",
      outer, inner, WARM_RUNS);

  print_block("General matrix (eigvals)", "Our alg.", wrap_eigvals, "DGEEV",
              wrap_dgeev, Ag, outer, inner);
  return 0;
}
