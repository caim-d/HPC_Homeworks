/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/

// moved parallel region inside dotprod and moved tid inside new parallel region, also added return statement

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod (float sum)
{
int i,tid;
// float sum;

#pragma omp parallel for reduction(+:sum) private(tid) schedule(static,1)
  for (i=0; i < VECLEN; i++)
    {
    tid = omp_get_thread_num();
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }

return sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

// #pragma omp parallel shared(sum)
  sum = dotprod(sum);

printf("Sum = %f\n",sum);

}
