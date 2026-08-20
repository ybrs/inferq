// Measures the cost of per-row DeltaNet state snapshots for inferq.
// Layout mirrors Qwen3.6-35B-A3B: 30 linear layers, per layer
//   recurrent = 32 * 128 * 128 f32 = 2 MiB
//   conv      = 8192 * 3 f32       = 96 KiB
// Access order mirrors the real forward: layers outer, rows inner.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define LAYERS 30
#define REC_BYTES (32UL * 128 * 128 * 4)
#define CONV_BYTES (8192UL * 3 * 4)

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void touch(float *p, size_t bytes) {
    size_t n = bytes / 4;
    for (size_t i = 0; i < n; i += 16) p[i] = p[i] * 0.999f + 0.001f;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 8;
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    float *rec[LAYERS], *conv[LAYERS];
    float *rec_snap[LAYERS], *conv_snap[LAYERS];
    for (int l = 0; l < LAYERS; l++) {
        rec[l] = aligned_alloc(64, REC_BYTES);
        conv[l] = aligned_alloc(64, CONV_BYTES);
        rec_snap[l] = aligned_alloc(64, REC_BYTES * rows);
        conv_snap[l] = aligned_alloc(64, CONV_BYTES * rows);
        memset(rec[l], 1, REC_BYTES);
        memset(conv[l], 1, CONV_BYTES);
        memset(rec_snap[l], 0, REC_BYTES * rows);
        memset(conv_snap[l], 0, CONV_BYTES * rows);
    }
    double best_copy = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        double copy_total = 0;
        for (int l = 0; l < LAYERS; l++) {
            for (int r = 0; r < rows; r++) {
                touch(rec[l], REC_BYTES);  // stand-in for the recurrence step
                double t0 = now();
                if (threads <= 1) {
                    memcpy((char *)rec_snap[l] + (size_t)r * REC_BYTES, rec[l], REC_BYTES);
                    memcpy((char *)conv_snap[l] + (size_t)r * CONV_BYTES, conv[l], CONV_BYTES);
                } else {
                    char *dst = (char *)rec_snap[l] + (size_t)r * REC_BYTES;
                    char *src = (char *)rec[l];
                    size_t chunk = REC_BYTES / threads;
#pragma omp parallel for num_threads(threads)
                    for (int t = 0; t < threads; t++) {
                        size_t off = (size_t)t * chunk;
                        size_t len = (t == threads - 1) ? REC_BYTES - off : chunk;
                        memcpy(dst + off, src + off, len);
                    }
                    memcpy((char *)conv_snap[l] + (size_t)r * CONV_BYTES, conv[l], CONV_BYTES);
                }
                copy_total += now() - t0;
            }
        }
        if (copy_total < best_copy) best_copy = copy_total;
    }
    double per_row = best_copy / rows;
    double bytes_per_row = (double)LAYERS * (REC_BYTES + CONV_BYTES);
    printf("rows=%d threads=%d  snapshot bytes/row=%.1f MiB  total=%.2f ms  per-row=%.3f ms  eff=%.2f GB/s\n",
           rows, threads, bytes_per_row / (1024 * 1024), best_copy * 1e3, per_row * 1e3,
           bytes_per_row * rows / best_copy / 1e9);
    return 0;
}
