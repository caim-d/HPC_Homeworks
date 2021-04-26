#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
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

void scan_mpi(long* prefix_sum, const long* A_local, long n_local, int mpirank, int mpisize) {
  if (n_local == 0) return;

  long corrections[mpisize]{};

  long* offset = (long*) malloc(mpisize * sizeof(long));
  long* prefix_sum_local = (long*) malloc(n_local * sizeof(long));

  prefix_sum_local[0] = A_local[0];
  for (long i = 1; i < n_local; i++) {
    prefix_sum_local[i] = prefix_sum_local[i-1] + A_local[i];
  }
  MPI_Barrier(MPI_COMM_WORLD);

  long offset_local = prefix_sum_local[n_local-1];

  MPI_Allgather(&offset_local, 1, MPI_LONG, offset, 1, MPI_LONG, MPI_COMM_WORLD);

  corrections[0] = 0;
  for (long j=1; j<mpisize; j++) {
    corrections[j] = corrections[j-1] + offset[j-1];
  }

  for (long j = 0; j < n_local; j++) {
    prefix_sum_local[j] = prefix_sum_local[j] + corrections[mpirank];
  }

  MPI_Gather(prefix_sum_local, n_local, MPI_LONG,
               prefix_sum, n_local, MPI_LONG, 0, MPI_COMM_WORLD);

  // if (mpirank == 0) {
  //   printf("FINAL prefix_sum = %d %d %d %d", prefix_sum[0], prefix_sum[1], prefix_sum[2], prefix_sum[3]);
  // }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int mpirank, mpisize;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);

  long N = 100000000; //4
  long N_local = N / mpisize;
  double tt;

  long* A = (long*) malloc(N * sizeof(long));
  long* A_local = (long*) malloc(N_local * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));


  if (mpirank == 0) {
    for (long i = 0; i < N; i++) A[i] = rand(); //i
  }

  MPI_Barrier(comm);

  MPI_Scatter(A, N_local, MPI_LONG, A_local, N_local, MPI_LONG, 0, comm);

  MPI_Barrier(comm);

  printf("Done 1 \n");

  if (mpirank == 0) {
  tt = MPI_Wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", MPI_Wtime() - tt);
  }

  MPI_Barrier(comm);
  tt = MPI_Wtime();
  printf("Done 2 \n");
  scan_mpi(B1, A_local, N_local, mpirank, mpisize);

  if (mpirank == 0) {
    printf("parallel-scan   = %fs\n", MPI_Wtime() - tt);

    long err = 0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);
  }

  free(A);
  free(A_local);
  free(B0);
  free(B1);


  MPI_Finalize();
  return 0;
}
