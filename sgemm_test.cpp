#include <iostream>
#include <sstream>
#include <cstdlib>
#include "timer.h"
#include "mkl.h"

void sgemm(bool aTranspose, bool bTranspose, float *A, float *B, float *C, int m, int n, int k, int lda, int ldb, int ldc) {
    float alpha = 1;
    float beta = 0;
    cblas_sgemm(CblasRowMajor, 
                aTranspose ? CblasTrans : CblasNoTrans,
                bTranspose ? CblasTrans : CblasNoTrans, 
                m, n, k, alpha,
                A, lda, 
                B, ldb, beta,
                C, ldc);
}

static void init(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 1.0f * rand() / RAND_MAX;
    }
}

int main(int argc, char **argv) {
    if (argc != 10) {
        printf("Usage: %s T/N T/N m n k lda ldb ldb loops\n", argv[0]);
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

    // Exclude the first time running
    sgemm(aTranspose, bTranspose, A, B, C, m, n, k, lda, ldb, ldc);

    std::ostringstream oss;
    oss << "Time of " << loops << " loops";
    std::string str = oss.str();
    {
        Timer t(str.c_str());
        for (int i = 0; i < loops; ++i) {
            sgemm(aTranspose, bTranspose, A, B, C, m, n, k, lda, ldb, ldc);
        }
    }

    return 0;
}
