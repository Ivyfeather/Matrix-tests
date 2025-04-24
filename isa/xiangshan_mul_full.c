#include "utils.h"
#include <riscv_matrix.h>
#include <stdlib.h> // Add header file to use malloc

/* Matrix Size */
#define M 512 // Rows of matrix A
#define K 7168 // Columns of matrix A and rows of matrix B
#define N 1024 // Columns of matrix B, this should be 4096, tmp try 1024 for shorter trace

/* Matrix Per Core Size */
#define M_PERCORE 256
#define K_PERCORE 7168
#define N_PERCORE 512

/* ----------------------------------------
  if K_PERCORE is too large so that A/B PERCORE cannot be loaded into L2 at the same time,
    we need to 

---------------------------------------- */
#define K_ONCE 512

#define L2_Banks 8

/* ----------------------------------------
  if K/N is not coprime with L2_Banks,
    we need to add padding to ensure matrix loads of rows are equally distributed among L2 banks
    (otherwise requests will flush into the same bank)
---------------------------------------- */
// TMP: we hard-code here
const int M_padding = M;
// since matrix A's row is 256B (4 lines in a row), we add 256B padding to K,
//   to ensure bankIdx is contiguously incremented
const int K_padding = K + 256;
const int N_padding = N + 64;

/* these are actually set in NEMU-Matrix/src/isa/riscv64/instr/rvmatrix/mreg.h */
// #define tmmax 64
// #define tkmax 256
// #define tnmax 64

/* ----------------------------------------
  large matrices as global, to avoid stack overflow, or use the following:
    int (*A)[K_padding] = aligned_alloc(64, M_padding * K_padding * sizeof(int8_t));
---------------------------------------- */ 
__attribute__((aligned(64))) int8_t A[M_padding][K_padding]; // Matrix A
__attribute__((aligned(64))) int8_t B[K_padding][N_padding]; // Matrix B
__attribute__((aligned(64))) int8_t C[M_padding][N_padding]; // Result matrix C
//TODO: C matrix is 32
static void test_xiangshan_mm() {
    int tile_m, tile_k, tile_n;
    msettype(E8, M1, BA);

    // Initialize matrices A, B, and C with some values (for testing)
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
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
        }
    }

    // Perform block matrix multiplication
    // outer loop for cores
    for (int m_outer = 0; m_outer < M/M_PERCORE; m_outer ++) {
        for (int n_outer = 0; n_outer < N/N_PERCORE; n_outer ++) {
            int m_base = m_outer * M_PERCORE;
            int n_base = n_outer * N_PERCORE;

            for (int k_outer = 0; k_outer < K/K_PERCORE; k_outer ++) {
                int k_base = k_outer * K_PERCORE;

                // inner loop for per core
                // we focus on how to calculate the result matrix C per core
                // size of C for each core is M_PERCORE × N_PERCORE
                for (int m = 0; m < M_PERCORE; m += tile_m) {
                    tile_m = msettilem(M_PERCORE - m);
                    for (int n = 0; n < N_PERCORE; n += tile_n) {
                        tile_n = msettilen(N_PERCORE - n);
                        mint8m1_t out = mlce8_m1(&C[m_base + m][n_base + n], N_padding * sizeof(int8_t));
                        
                        for (int k = 0; k < K_PERCORE; k += tile_k) {
                            tile_k = msettilek(K_PERCORE - k);
                            mint8m1_t tr_a = mlae8_m1(&A[m_base + m][k_base + k], K_padding * sizeof(int8_t));
                            mint8m1_t tr_b = mlbe8_m1(&B[k_base + k][n_base + n], N_padding * sizeof(int8_t));
                            out = mma_mm(out, tr_a, tr_b);
                        }
                        msce8_m(out, &C[m_base + m][n_base + n], N_padding * sizeof(int8_t));
                    }
                }
                // inner m × k × n done, proceed to next k_outer
                printf("@@ core %d, mi %d, ni %d, ki %d\n",
                    m_outer * N/N_PERCORE * K/K_PERCORE + n_outer * K/K_PERCORE + k_outer,
                    m_outer, n_outer, k_outer);
            }
            // proceed to next n_outer
            printf("@@ done C %d %d\n", m_outer, n_outer);
            
        }// TODO: add L2 cache load per segment
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
