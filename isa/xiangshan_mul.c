#include "utils.h"
#include <riscv_matrix.h>

#define M 256 // Rows of matrix A
#define K 512 // Columns of matrix A and rows of matrix B
#define N 512 // Columns of matrix B

// these are actually set in NEMU-Matrix/src/isa/riscv64/instr/rvmatrix/mreg.h
// unused
#define tmmax 64
#define tkmax 256
#define tnmax 64

#define L2_Banks 8

static void test_xiangshan_mm() {
    int tile_m, tile_k, tile_n;
    msettype(E8, M1, BA);

    // if K/N is not coprime with L2_Banks,
    //   we need to add padding to ensure matrix loads of rows are equally distributed among L2 banks
    //   (otherwise requests will flush into the same bank)
    // TMP: we hard-code here
    // TODO: no need for M, right?
    const int M_padding = M;
    // since mload.a row is 256B (4 lines in a row), we add 256B padding to K,
    //   to ensure bankIdx is contiguously incremented
    const int K_padding = K + 256;
    const int N_padding = N + 64;

    __attribute__((aligned(64))) int8_t A[M_padding][K_padding]; // Matrix A
    __attribute__((aligned(64))) int8_t B[K_padding][N_padding]; // Matrix B
    __attribute__((aligned(64))) int8_t C[M_padding][N_padding] = {0}; // Result matrix C

    // Initialize matrices A and B with some values (for testing)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = (i + j) % 128 - 64; // Example initialization
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = (i - j) % 128 - 64; // Example initialization
        }
    }

    // Perform block matrix multiplication
    for (int m = 0; m < M; m += tile_m) {
        tile_m = msettilem(M - m);

        for (int n = 0; n < N; n += tile_n) {
            tile_n = msettilen(N - n);
            mint8m1_t out = mlce8_m1(&C[m][n], N_padding * sizeof(int8_t));

            for (int k = 0; k < K; k += tile_k) {
                tile_k = msettilek(K - k);
                mint8m1_t tr_a = mlae8_m1(&A[m][k], K_padding * sizeof(int8_t));
                mint8m1_t tr_b = mlbe8_m1(&B[k][n], N_padding * sizeof(int8_t));
                out = mma_mm(out, tr_a, tr_b);
            }
            msce8_m(out, &C[m][n], N_padding * sizeof(int8_t));
        }
    }

    // Print result matrix C (optional, for verification)
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", C[i][j]);
    //     }
    //     printf("\n");
    // }
}

#define DISABLE_TIME_INTR 0x100
#define NOTIFY_PROFILER 0x101
#define NOTIFY_PROFILE_EXIT 0x102
#define GOOD_TRAP 0x0

void nemu_signal(int a){
    asm volatile ("mv a0, %0\n\t"
                  ".insn r 0x6B, 0, 0, x0, x0, x0\n\t"
                  :
                  : "r"(a)
                  : "a0");
}

int main() {
    printf("Hello, RISC-V World!\n");
    test_xiangshan_mm();
    printf("Matrix Multiplication Test Done\n");
    nemu_signal(GOOD_TRAP);
    return 0;
}
