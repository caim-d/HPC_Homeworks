#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];                              // changed this to start from A[0]
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;

  int tid;
  long count{};
  long p{8};
  omp_set_num_threads(p);
  long interval_length{(n/p) + 1};
  long corrections[p]{};

  #pragma omp parallel private(count)
  {
  count = 0;
  #pragma omp for schedule(static, interval_length)
  for (long j = 0; j < n; j++){
    if (count == 0) {
      prefix_sum[j] = A[j];
    }
    else {
      prefix_sum[j] = prefix_sum[j-1] + A[j];
    }
    count += 1;
  }
  }

  corrections[0] = 0;
  for (long j=1; j<p; j++) {
   corrections[j] = corrections[j-1] + prefix_sum[j*interval_length-1];
  }

  #pragma omp parallel for schedule(static, interval_length)
  for (long j=0; j<n; j++) {
    prefix_sum[j] += corrections[j/interval_length];
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
