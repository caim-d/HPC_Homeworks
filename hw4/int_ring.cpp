#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>


int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status status;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

		if (rank == 0) {
			printf("Size = %d \n", size);
		}

		double tt;

    int x_in;
    int x_out = 0;

    MPI_Barrier(comm);

		tt = MPI_Wtime();
    int N = 10000;
    for (int i=0; i<N; i++) {

    	if (rank == 0) { // initial send
            MPI_Send(&x_out, 1, MPI_INT, (rank+1)%size, 266, comm);
	    MPI_Recv(&x_in, 1, MPI_INT, (rank-1)%size, 266, comm, &status);
	    x_out = x_in;
        }
        else {
	    MPI_Recv(&x_in, 1, MPI_INT, (rank-1)%size, 266, comm, &status);
    	    x_out = x_in + rank;
    	    MPI_Send(&x_out, 1, MPI_INT, (rank+1)%size, 266, comm);
    	}
    }
		tt = MPI_Wtime() - tt;

		if (rank == 0) {
			printf("Final x = %d \n", x_in);
			printf("Latency = %e ms \n", tt/(N*size) * 1000);
		}

//------------------------------------------------------------------------------
		MPI_Barrier(comm);
		int* A_in = (int*) calloc(500000, sizeof(int));
    int* A_out = (int*) calloc(500000, sizeof(int));

		MPI_Barrier(comm);

		tt = MPI_Wtime();
    for (int i=0; i<N; i++) {

    	if (rank == 0) { // initial send
            MPI_Send(A_out, 500000, MPI_INT, (rank+1)%size, 266, comm);
	    MPI_Recv(A_in, 500000, MPI_INT, (rank-1)%size, 266, comm, &status);
        }
        else {
	    MPI_Recv(A_in, 500000, MPI_INT, (rank-1)%size, 266, comm, &status);
    	    MPI_Send(A_out, 500000, MPI_INT, (rank+1)%size, 266, comm);
    	}
    }
		tt = MPI_Wtime() - tt;

		if (rank == 0) {
			printf("Time = %e \n", tt);
			printf("Size = %d \n", size);
			printf("Bandwidth = %f Gb/s \n", 500000*sizeof(int)*(size*N)/tt/1e9);
		}

    MPI_Finalize();
    return 0;
}
