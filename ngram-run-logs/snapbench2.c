// Variant of snapbench with non-temporal (streaming) stores.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
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

static void nt_copy(char *dst, const char *src, size_t bytes) {
    size_t i = 0;
    for (; i + 128 <= bytes; i += 128) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(src + i + 32));
        __m256i c = _mm256_loadu_si256((const __m256i *)(src + i + 64));
        __m256i d = _mm256_loadu_si256((const __m256i *)(src + i + 96));
        _mm256_stream_si256((__m256i *)(dst + i), a);
        _mm256_stream_si256((__m256i *)(dst + i + 32), b);
        _mm256_stream_si256((__m256i *)(dst + i + 64), c);
        _mm256_stream_si256((__m256i *)(dst + i + 96), d);
    }
    if (i < bytes) memcpy(dst + i, src + i, bytes - i);
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 8;
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    float *rec[LAYERS], *conv[LAYERS], *rec_snap[LAYERS], *conv_snap[LAYERS];
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
    double best = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        double total = 0;
        for (int l = 0; l < LAYERS; l++) {
            for (int r = 0; r < rows; r++) {
                touch(rec[l], REC_BYTES);
                double t0 = now();
                char *dst = (char *)rec_snap[l] + (size_t)r * REC_BYTES;
                if (threads <= 1) {
                    nt_copy(dst, (char *)rec[l], REC_BYTES);
                } else {
                    size_t chunk = (REC_BYTES / threads) & ~127UL;
#pragma omp parallel for num_threads(threads)
                    for (int t = 0; t < threads; t++) {
                        size_t off = (size_t)t * chunk;
                        size_t len = (t == threads - 1) ? REC_BYTES - off : chunk;
                        nt_copy(dst + off, (char *)rec[l] + off, len);
                    }
                }
                nt_copy((char *)conv_snap[l] + (size_t)r * CONV_BYTES, (char *)conv[l], CONV_BYTES);
                _mm_sfence();
                total += now() - t0;
            }
        }
        if (total < best) best = total;
    }
    double bytes_per_row = (double)LAYERS * (REC_BYTES + CONV_BYTES);
    printf("NT rows=%d threads=%d  per-row=%.3f ms  total=%.2f ms  eff=%.2f GB/s\n",
           rows, threads, best / rows * 1e3, best * 1e3, bytes_per_row * rows / best / 1e9);
    return 0;
}
