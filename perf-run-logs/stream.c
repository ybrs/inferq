/* Minimal STREAM-like triad benchmark (McCalpin STREAM semantics), public domain style. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 80000000
#endif

static double a[STREAM_ARRAY_SIZE], b[STREAM_ARRAY_SIZE], c[STREAM_ARRAY_SIZE];

static double mysecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main() {
    int i;
    double scalar = 3.0;
    double times[4][10];
    int NTIMES = 10;

    #pragma omp parallel for
    for (i = 0; i < STREAM_ARRAY_SIZE; i++) {
        a[i] = 1.0; b[i] = 2.0; c[i] = 0.0;
    }

    int nthreads = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }
    #endif
    fprintf(stderr, "Threads: %d, Array size: %d elements (%.1f MB each)\n",
            nthreads, STREAM_ARRAY_SIZE, STREAM_ARRAY_SIZE * 8.0 / 1024.0 / 1024.0);

    for (int k = 0; k < NTIMES; k++) {
        double t;
        t = mysecond();
        #pragma omp parallel for
        for (i = 0; i < STREAM_ARRAY_SIZE; i++) c[i] = a[i];
        times[0][k] = mysecond() - t;

        t = mysecond();
        #pragma omp parallel for
        for (i = 0; i < STREAM_ARRAY_SIZE; i++) b[i] = scalar * c[i];
        times[1][k] = mysecond() - t;

        t = mysecond();
        #pragma omp parallel for
        for (i = 0; i < STREAM_ARRAY_SIZE; i++) c[i] = a[i] + b[i];
        times[2][k] = mysecond() - t;

        t = mysecond();
        #pragma omp parallel for
        for (i = 0; i < STREAM_ARRAY_SIZE; i++) a[i] = b[i] + scalar * c[i];
        times[3][k] = mysecond() - t;
    }

    const char *label[4] = {"Copy", "Scale", "Add", "Triad"};
    double bytes[4] = {
        2.0 * sizeof(double) * STREAM_ARRAY_SIZE,
        2.0 * sizeof(double) * STREAM_ARRAY_SIZE,
        3.0 * sizeof(double) * STREAM_ARRAY_SIZE,
        3.0 * sizeof(double) * STREAM_ARRAY_SIZE
    };

    printf("Function    Best Rate MB/s   Avg time     Min time     Max time\n");
    for (int j = 0; j < 4; j++) {
        double mint = FLT_MAX, avgt = 0, maxt = 0;
        for (int k = 1; k < NTIMES; k++) { /* skip first iteration */
            avgt += times[j][k];
            if (times[j][k] < mint) mint = times[j][k];
            if (times[j][k] > maxt) maxt = times[j][k];
        }
        avgt /= (NTIMES - 1);
        printf("%-11s %12.1f  %11.6f  %11.6f  %11.6f\n",
               label[j], 1.0E-06 * bytes[j] / mint, avgt, mint, maxt);
    }
    /* prevent dead-code elimination */
    if (a[1] + b[1] + c[1] == -1.0) printf("dummy\n");
    return 0;
}
