#include "utils.h"
#include <riscv_matrix.h>

static void test_xiangshan_mm() {
    const int M = 128; // Rows of matrix A
    const int K = 512; // Columns of matrix A and rows of matrix B
    const int N = 512; // Columns of matrix B
    const int tmmax = 64;
    const int tkmax = 256;
    const int tnmax = 64;
    int tile_m, tile_k, tile_n;
    msettype(E8, M1, BA);

    int8_t A[M][K]; // Matrix A
    int8_t B[K][N]; // Matrix B
    int8_t C[M][N] = {0}; // Result matrix C initialized to 0

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
            mint8m1_t out = mlce8_m1(&C[m][n], N * sizeof(int8_t));

            for (int k = 0; k < K; k += tile_k) {
                tile_k = msettilek(K - k);
                mint8m1_t tr_a = mlae8_m1(&A[m][k], K * sizeof(int8_t));
                mint8m1_t tr_b = mlbe8_m1(&B[k][n], N * sizeof(int8_t));
                printf("@@ m: %d, k: %d, n: %d\n", m, k, n);
                printf("  @@ tile_m: %d, tile_k: %d, tile_n: %d\n", tile_m, tile_k, tile_n);
                out = mma_mm(out, tr_a, tr_b);
            }
            msce8_m(out, &C[m][n], N * sizeof(int8_t));
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
