#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "generated_matrix_config.h"

/* Matrix Size */
#define M MATRIX_M
#define K MATRIX_K
#define N MATRIX_N

#define CACHE_LINE_BYTES 64ULL
#define BLOCK_M 64
#define BLOCK_K 256
#define BLOCK_N 64

typedef struct {
    uint64_t a_base;
    uint64_t b_base;
    uint64_t c_base;
} matrix_layout_t;

static uint64_t align_up_u64(uint64_t value, uint64_t align) {
    return ((value + align - 1ULL) / align) * align;
}

static matrix_layout_t build_layout(void) {
    matrix_layout_t layout;
    uint64_t a_size = (uint64_t)M * (uint64_t)K * sizeof(int8_t);
    uint64_t b_size = (uint64_t)K * (uint64_t)N * sizeof(int8_t);

    layout.a_base = 0ULL;
    layout.b_base = align_up_u64(layout.a_base + a_size, CACHE_LINE_BYTES);
    layout.c_base = align_up_u64(layout.b_base + b_size, CACHE_LINE_BYTES);
    return layout;
}

static void trace_block(char rw, uint64_t base_addr, int rows, uint64_t row_stride_bytes,
                        uint64_t block_row_bytes, char matrix_type) {
    for (int row = 0; row < rows; row++) {
        uint64_t row_base = base_addr + (uint64_t)row * row_stride_bytes;
        for (uint64_t offset = 0; offset < block_row_bytes; offset += CACHE_LINE_BYTES) {
            printf("%c%c 0x%llx %c\n", rw == 'r' ? 'M' : 'M', rw == 'r' ? 'R' : 'W',
                   (unsigned long long)(row_base + offset), matrix_type);
        }
    }
}

int main(void) {
    matrix_layout_t layout = build_layout();
    uint64_t checksum = 0;

    if (M % BLOCK_M != 0 || K % BLOCK_K != 0 || N % BLOCK_N != 0) {
        fprintf(stderr, "M/K/N must be multiples of 64/256/64, got %d/%d/%d\n", M, K, N);
        return 1;
    }

    printf("# M=%d K=%d N=%d\n", M, K, N);
    printf("# block=%d x %d x %d\n", BLOCK_M, BLOCK_K, BLOCK_N);
    printf("# A_BASE=%llu B_BASE=%llu C_BASE=%llu\n",
           (unsigned long long)layout.a_base,
           (unsigned long long)layout.b_base,
           (unsigned long long)layout.c_base);

    for (int mb = 0; mb < M; mb += BLOCK_M) {
        for (int nb = 0; nb < N; nb += BLOCK_N) {
            uint64_t c_base = layout.c_base + ((uint64_t)mb * (uint64_t)N + (uint64_t)nb) * sizeof(int32_t);

            // trace_block('r', c_base, BLOCK_M, (uint64_t)N * sizeof(int32_t), BLOCK_N * sizeof(int32_t), 'c');

            for (int kb = 0; kb < K; kb += BLOCK_K) {
                uint64_t a_base = layout.a_base + (uint64_t)mb * (uint64_t)K + (uint64_t)kb;
                uint64_t b_base = layout.b_base + (uint64_t)kb * (uint64_t)N + (uint64_t)nb;

                trace_block('r', a_base, BLOCK_M, (uint64_t)K * sizeof(int8_t), BLOCK_K * sizeof(int8_t), 'a');
                trace_block('r', b_base, BLOCK_K, (uint64_t)N * sizeof(int8_t), BLOCK_N * sizeof(int8_t), 'b');

                checksum ^= (uint64_t)(mb + 1) * (uint64_t)(nb + 1) * (uint64_t)(kb + 1);
            }

            trace_block('w', c_base, BLOCK_M, (uint64_t)N * sizeof(int32_t), BLOCK_N * sizeof(int32_t), 'c');
        }
    }

    printf("# trace_summary checksum=%llu\n", (unsigned long long)checksum);
    return 0;
}
