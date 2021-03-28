#include <iostream>
#include <cmath>
#include "utils.h"
#include <omp.h>

double *gauss_seidel_iterate(double *u, double *u_new, double *f, double h2, int n);
double get_norm(double *u, double *f, double h2, int n);

int main() {
  const int n{100};
  const double h{1.0/(n+1)};
  const double h2{h*h};
  int tid;

  double *u = (double*) calloc((n+2)*(n+2), sizeof(double));
  double *f = (double*) malloc((n+2)*(n+2) * sizeof(double));

  double *u_new = (double*) calloc((n+2)*(n+2) , sizeof(double));

  for (int i = 0; i<(n+2)*(n+2); i++) {
    f[i] = 1;
  }

  double residual_norm{get_norm(u, f, h2, n)};
  std::cout << "Initial ||Au-f|| = " << residual_norm << '\n';

  double threshold{residual_norm * 1e-6};
  int counter{};

  Timer t;
  t.tic();

  while (residual_norm >= threshold) {
    if (counter >= 5000) {
      break;
    }
    else {
      double sum_sqrt{0};
      #pragma omp parallel num_threads(1)
      {
        u = gauss_seidel_iterate(u, u_new, f, h2, n);

        // calculate residual norm without function call to do reduction
        #pragma omp for reduction(+:sum_sqrt)
        for (int i=1; i<n+1; i++) {
          for (int j=1; j<n+1; j++) {
            double sum_Au_ij{
              1/h2 * (4*u[i+j*(n+2)] - u[i-1+j*(n+2)] - u[i+1+j*(n+2)] - u[i+(j-1)*(n+2)] -
              u[i+(j+1)*(n+2)])
            };
          sum_sqrt += pow(sum_Au_ij - f[i+j*(n+2)], 2);
          }
        }
      }
      residual_norm = sqrt(sum_sqrt);

      counter += 1;
      if (counter%100 == 0) {
        printf("After iteration %d, ||Au-f|| = %f. \n", counter, residual_norm);
      }
    }
  }
  double time = t.toc();
  std::cout << "Time taken = " << time << '\n';

  free(u);
  free(f);
  free(u_new);

  return 0;
};
//------------------------------------------------------------------------------

double *gauss_seidel_iterate(double *u, double *u_new, double *f, double h2, int n) {
  // red points (odd numbers)
  #pragma omp for
  for (int i=1; i<n+1; i++) {
    for (int j=1; j<n+1; j++) {
      if (i%2 == j%2){ // red points
        u_new[i+j*(n+2)] = 0.25 * (h2*f[i+j*(n+2)] + u[i-1+j*(n+2)] + u[i+(j-1)*(n+2)] +
                                             u[i+1+j*(n+2)] + u[i+(j+1)*(n+2)]);
      }
    }
  }
  #pragma omp for
  for (int i=1; i<n+1; i++) {
    for (int j=1; j<n+1; j++) {
      if (i%2 == (j+1)%2) { // black points
        u_new[i+j*(n+2)] = 0.25 * (h2*f[i+j*(n+2)] + u_new[i-1+j*(n+2)] + u_new[i+(j-1)*(n+2)] +
                                           u_new[i+1+j*(n+2)] + u_new[i+(j+1)*(n+2)]);
      }
    }
  }

  // set output equal to u_new
  #pragma omp for
  for (int i=1; i<n+1; i++) {
    for (int j=1; j<n+1; j++) {
      u[i+j*(n+2)] = u_new[i+j*(n+2)];
    }
  }
  return u;
}

// calculate l2 norm of Au-f
double get_norm(double *u, double *f, double h2, int n) {
  double sum_sqrt{0};
  for (int i=1; i<n+1; i++) {
    for (int j=1; j<n+1; j++) {
      double sum_Au_ij{
        1/h2 * (4*u[i+j*(n+2)] - u[i-1+j*(n+2)] - u[i+1+j*(n+2)] - u[i+(j-1)*(n+2)] -
        u[i+(j+1)*(n+2)])
      };
      sum_sqrt += pow(sum_Au_ij - f[i+j*(n+2)], 2);
    }
  }
  return sqrt(sum_sqrt);
}
