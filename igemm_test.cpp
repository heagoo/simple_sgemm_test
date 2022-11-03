#include <emmintrin.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "mkl.h"
#include "timer.h"

void igemm(bool aTranspose, bool bTranspose, int8_t *A, int8_t *B, int32_t *C, int m, int n, int k, int lda, int ldb, int ldc) {
    float alpha = 1;
    float beta = 0;
    int oc = 0;

    cblas_gemm_s8u8s32(CblasRowMajor,
                       aTranspose ? CblasTrans : CblasNoTrans,
                       bTranspose ? CblasTrans : CblasNoTrans,
                       CblasFixOffset, 
                       m, n, k, alpha, 
                       A, lda, 0, 
                       B, ldb, 0, beta, 
                       C, ldc, &oc);
}

template <typename T>
static void init(T *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (T) (127.0f * rand() / RAND_MAX);
    }
}

#define CACHELINE_SIZE 64
template <typename T>
static void flush_cache(const T *buf, size_t size) {
#pragma omp parallel for
    for (size_t offset = 0; offset < size; offset += CACHELINE_SIZE / sizeof(T)) {
        _mm_clflush(buf + offset);
    }
}

int main(int argc, char **argv) {
    if (argc < 10) {
        printf("Usage: %s T/N T/N m n k lda ldb ldc loops [flush_b=0]\n", argv[0]);
        exit(-1);
    }

    bool aTranspose = (argv[1][0] == 'T');
    bool bTranspose = (argv[2][0] == 'T');
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    int k = atoi(argv[5]);
    int lda = atoi(argv[6]);
    int ldb = atoi(argv[7]);
    int ldc = atoi(argv[8]);
    int loops = atoi(argv[9]);
    int flush_b = 0;
    if (argc > 10) {
        flush_b = atoi(argv[10]);
    }

    int sizeA = m * lda;
    if (aTranspose) {
        sizeA = k * lda;
    }
    int8_t *A = new int8_t[sizeA];

    int sizeB = k * ldb;
    if (bTranspose) {
        sizeB = n * ldb;
    }
    int8_t *B = new int8_t[sizeB];

    int sizeC = m * ldc;
    int32_t *C = new int32_t[sizeC];

    init(A, sizeA);
    init(B, sizeB);

    // Exclude the first time running
    igemm(aTranspose, bTranspose, A, B, C, m, n, k, lda, ldb, ldc);

    float totalTime = 0;
    for (int i = 0; i < loops; ++i) {
        if (flush_b) {
            flush_cache(B, sizeB);
        }

        Timer t;
        igemm(aTranspose, bTranspose, A, B, C, m, n, k, lda, ldb, ldc);
        totalTime += t.getTime();
    }

    printf("Average %f ms per igemm(m=%d,n=%d,k=%d)\n", totalTime / loops, m, n, k);

    return 0;
}
