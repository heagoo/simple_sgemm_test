#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include "timer.h"
#include "mkl.h"

void packed_sgemm(bool aTranspose, bool bTranspose, float *A, float *Bp, float *C, int m, int n, int k, int lda, int ldb, int ldc) {
    float beta = 0;

    // SGEMM computations are performed using the packed B matrix: Bp
    auto transa = (aTranspose ? CblasTrans : CblasNoTrans);
    cblas_sgemm_compute(CblasRowMajor, transa, CblasPacked, m, n, k, A, lda, Bp, ldb, beta, C, ldc);
}

static void init(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 1.0f * rand() / RAND_MAX;
    }
}

int main(int argc, char **argv) {
    if (argc != 10) {
        printf("Usage: %s T/N T/N m n k lda ldb ldc loops\n", argv[0]);
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

    int sizeA = m * lda;
    if (aTranspose) { sizeA = k * lda; }
    float *A = new float[sizeA];

    int sizeB = k * ldb;
    if (bTranspose) { sizeB = n * ldb; }
    float *B = new float[sizeB];

    int sizeC = m * ldc;
    float *C = new float[sizeC];

    init(A, sizeA);
    init(B, sizeB);
    init(C, sizeC);

    // allocate memory for packed data format
    size_t Bp_size = cblas_sgemm_pack_get_size(CblasBMatrix, m, n, k);
    float *Bp = (float *)mkl_malloc(Bp_size, 64);
    
    // transform B into packed format
    float alpha = 1;
    auto transb = (bTranspose ? CblasTrans : CblasNoTrans);
    cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, transb, m, n, k, alpha, B, ldb, Bp);

    // Warm up
    packed_sgemm(aTranspose, bTranspose, A, Bp, C, m, n, k, lda, ldb, ldc);

    std::ostringstream oss;
    oss << "Time of " << loops << " loops";
    std::string str = oss.str();
    {
        Timer t(str.c_str());
        for (int i = 0; i < loops; ++i) {
            packed_sgemm(aTranspose, bTranspose, A, Bp, C, m, n, k, lda, ldb, ldc);
        }
    }

    // release the memory for Bp
    mkl_free(Bp);

    return 0;
}
