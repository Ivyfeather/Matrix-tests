#include "utils.h"
#include <riscv_matrix.h>
#include <stdlib.h> // Add header file to use malloc
#include <assert.h>

#define SMALL
/* Matrix Size */
#ifdef SMALL
    #define M 64 // Columns of matrix B, this should be 512, tmp try for shorter trace
    #define K 256 // Columns of matrix A and rows of matrix B, this should be 7168
    #define N 64 // Columns of matrix B, this should be 4096
#else
    /* Matrix Size */
    #define M 512 // Columns of matrix B, this should be 512, tmp try for shorter trace
    #define K 7168 // Columns of matrix A and rows of matrix B, this should be 7168
    #define N 4096 // Columns of matrix B, this should be 4096
#endif

#define SINGLE_CORE
/* Matrix Per Core Size */
#ifdef SINGLE_CORE
    #define M_PERCORE M
    #define K_PERCORE K
    #define N_PERCORE N
#else
    #define M_PERCORE 256
    #define K_PERCORE 7168
    #define N_PERCORE 512
#endif

/* ----------------------------------------
  if K_PERCORE is too large so that A/B PERCORE cannot be loaded into L2 at the same time,
    we need to load part by part of A (M_PERCORE × K_ONCE) and B (K_ONCE × N_PERCORE) into L2,
    to achieve better data reuse in L2
---------------------------------------- */
#define K_ONCE 256

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
__attribute__((aligned(64))) int32_t C[M_padding][N_padding]; // Result matrix C

/* ----------------------------------------
    NEMU TRAP signals
---------------------------------------- */
#define DISABLE_TIME_INTR 0x100
#define NOTIFY_PROFILER 0x101
#define NOTIFY_PROFILE_EXIT 0x102
#define GOOD_TRAP 0x0
#define TRACE_DUMP 0x103
#define TRACE_END 0x104

void nemu_signal(int a){
    asm volatile ("mv a0, %0\n\t"
                  ".insn r 0x6B, 0, 0, x0, x0, x0\n\t"
                  :
                  : "r"(a)
                  : "a0");
}

/* ----------------------------------------
    MMA test function
---------------------------------------- */
static int test_xiangshan_mm() {
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

    // Assert that dimensions are properly divisible
    assert(M % M_PERCORE == 0 && "M must be divisible by M_PERCORE");
    assert(N % N_PERCORE == 0 && "N must be divisible by N_PERCORE");
    assert(K % K_PERCORE == 0 && "K must be divisible by K_PERCORE");
    assert(K_PERCORE % K_ONCE == 0 && "K_PERCORE must be divisible by K_ONCE");

    // Perform block matrix multiplication
    // outer loop for cores
    for (int m_outer = 0; m_outer < M/M_PERCORE; m_outer ++) {
        for (int n_outer = 0; n_outer < N/N_PERCORE; n_outer ++) {
            int m_base = m_outer * M_PERCORE;
            int n_base = n_outer * N_PERCORE;

            for (int k_outer = 0; k_outer < K/K_PERCORE; k_outer ++) {
                int k_outer_base = k_outer * K_PERCORE;

                // inner loop for per core
                // we focus on how to calculate the result matrix C per core
                // size of C for each core is M_PERCORE × N_PERCORE

                for (int k_inner = 0; k_inner < K_PERCORE/K_ONCE; k_inner ++) {
                    int k_inner_base = k_outer_base + k_inner * K_ONCE;

                    for (int m = 0; m < M_PERCORE; m += tile_m) {
                        tile_m = msettilem(M_PERCORE - m);
                        for (int n = 0; n < N_PERCORE; n += tile_n) {
                            tile_n = msettilen(N_PERCORE - n);
                            msettype(E32, M4, BA);
                            mint32m4_t tr_c = mlce32_m4(&C[m_base + m][n_base + n], N_padding * sizeof(int32_t));
                            msettype(E8, M1, BA);

                            for (int k = 0; k < K_ONCE; k += tile_k) {
                                tile_k = msettilek(K_ONCE - k);
                                mint8m1_t tr_a = mlae8_m1(&A[m_base + m][k_inner_base + k], K_padding * sizeof(int8_t));
                                mint8m1_t tr_b = mlbe8_m1(&B[k_inner_base + k][n_base + n], N_padding * sizeof(int8_t));
                                tr_c = mqma_mm(tr_c, tr_a, tr_b);
                            }
                            msettype(E32, M4, BA);
                            msce32_m(tr_c, &C[m_base + m][n_base + n], N_padding * sizeof(int32_t));
                            msettype(E8, M1, BA);
                        }
                    }
                    // inner m × k × n done, proceed to next k_outer
                    printf("-- core %d, mi %d, ni %d, ki %d, kseg %d\n",
                        m_outer * N/N_PERCORE * K/K_PERCORE + n_outer * K/K_PERCORE + k_outer,
                        m_outer, n_outer, k_outer, k_inner);

                }
                // inner m × k × n done, proceed to next k_outer
                printf("@@ core %d, mi %d, ni %d, ki %d\n",
                    m_outer * N/N_PERCORE * K/K_PERCORE + n_outer * K/K_PERCORE + k_outer,
                    m_outer, n_outer, k_outer);
            }
            // proceed to next n_outer
            printf("== done C %d %d\n", m_outer, n_outer);
            
        }// TODO: add L2 cache load per segment
    }

    // Software-Check result matrix C
    // only check a portion, cause it takes too long to check all
    /*
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            int32_t result = 0;
            for (int k = 0; k < K; k++) {
                result += (int32_t)A[i][k] * (int32_t)B[k][j];
            }

            if (C[i][j] != result) {
                printf("Mismatch at C[%d][%d]: expected %d, got %d\n", i, j, result, C[i][j]);
                return 1;
            }
        }
    }
    */

    // Hardware-Check: load data again through DCache, for tl-test to check
    int32_t sum = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            sum += C[i][j];
        }
    }
    printf("sum of all elements in C: %d\n", sum);

    // use mlce32_m4 to set matrix C to random values, to test DCache read
    for (int m = 0; m < M; m += tile_m) {
        for (int n = 0; n < N; n += tile_n) {
            msettype(E32, M4, BA);
            mint32m4_t tr_c; // TODO: may cause pointer issue, but hardware just needs address
            msce32_m(tr_c, &C[m][n], N_padding * sizeof(int32_t));
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            sum += C[i][j];
        }
    }
    printf("[PASS2] sum of all elements in C: %d\n", sum);

    return 0;
}

int main() {
    printf("Hello, RISC-V World!\n");
    printf("Starting xiangshan matrix multiplication test...\n");

    nemu_signal(TRACE_DUMP);
    int result = test_xiangshan_mm();
    nemu_signal(TRACE_END);

    printf("Matrix Multiplication Test Done\n");
    nemu_signal(result);
    return 0;
}
