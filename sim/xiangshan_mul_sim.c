#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "generated_matrix_config.h"

/* Matrix Size */
#define M MATRIX_M
#define K MATRIX_K
#define N MATRIX_N

#define CACHE_LINE_BYTES 64ULL
#define L2_SETS 512
#define L2_WAYS 8
#define L2_COLOR_BYTES (L2_SETS * CACHE_LINE_BYTES)
#define BLOCK_M 64
#define BLOCK_K 256
#define BLOCK_N 32
#ifndef MATRIX_C_STRIP_M
#define MATRIX_C_STRIP_M 16
#endif
#define C_STRIP_M MATRIX_C_STRIP_M
#ifndef MATRIX_REREAD_B_PER_STRIP
#define MATRIX_REREAD_B_PER_STRIP 0
#endif
#ifndef MATRIX_EMIT_REUSE_READS
#define MATRIX_EMIT_REUSE_READS 1
#endif

typedef struct {
    uint64_t a_base;
    uint64_t b_base;
    uint64_t c_base;
} matrix_layout_t;

typedef struct {
    int bm;
    int bk;
    int bn;
    uint64_t working_set_bytes;
    uint64_t big_block_count;
    uint64_t kernels_per_block;
    uint64_t a_reuse_per_block;
    uint64_t b_reuse_per_block;
    uint64_t total_reuse_per_block;
    uint64_t total_reuse;
} block_scheme_t;

static uint64_t align_up_u64(uint64_t value, uint64_t align) {
    return ((value + align - 1ULL) / align) * align;
}

static matrix_layout_t build_layout(void) {
    matrix_layout_t layout;
    uint64_t a_size = (uint64_t)M * (uint64_t)K * sizeof(int8_t);
    uint64_t b_size = (uint64_t)K * (uint64_t)N * sizeof(int8_t);

    layout.a_base = 0ULL;
    /*
     * Color A/B/C so a selected A+B+C block can fit within 8 L2 ways. B starts
     * at a color boundary; C starts one line after a color boundary to avoid
     * the densest overlap between B rows and C rows.
     */
    layout.b_base = align_up_u64(layout.a_base + a_size, L2_COLOR_BYTES);
    layout.c_base = align_up_u64(layout.b_base + b_size, L2_COLOR_BYTES) + CACHE_LINE_BYTES;
    return layout;
}

static void trace_range(char rw, uint64_t base_addr, uint64_t bytes, char matrix_type) {
    for (uint64_t offset = 0; offset < bytes; offset += CACHE_LINE_BYTES) {
        printf("M%c 0x%llx %c\n", rw == 'r' ? 'R' : 'W',
               (unsigned long long)(base_addr + offset), matrix_type);
    }
}

static void trace_matrix_block(char rw, uint64_t base_addr, int rows, uint64_t row_stride_bytes,
                               uint64_t row_bytes, char matrix_type) {
    for (int row = 0; row < rows; row++) {
        uint64_t row_base = base_addr + (uint64_t)row * row_stride_bytes;
        trace_range(rw, row_base, row_bytes, matrix_type);
    }
}

static void fill_scheme_stats(block_scheme_t *scheme) {
    uint64_t blocks_m = (uint64_t)M / (uint64_t)scheme->bm;
    uint64_t blocks_k = (uint64_t)K / (uint64_t)scheme->bk;
    uint64_t blocks_n = (uint64_t)N / (uint64_t)scheme->bn;

    scheme->big_block_count = blocks_m * blocks_k * blocks_n;

    uint64_t tiles_m = (uint64_t)scheme->bm / BLOCK_M;
    uint64_t tiles_k = (uint64_t)scheme->bk / BLOCK_K;
    uint64_t tiles_n = (uint64_t)scheme->bn / BLOCK_N;

    scheme->kernels_per_block = tiles_m * tiles_k * tiles_n;

    uint64_t a_tiles_per_block = tiles_m * tiles_k;
    uint64_t b_tiles_per_block = tiles_k * tiles_n;
    scheme->a_reuse_per_block = a_tiles_per_block * (tiles_n - 1ULL);
    scheme->b_reuse_per_block = b_tiles_per_block * (tiles_m - 1ULL);
    scheme->total_reuse_per_block = scheme->a_reuse_per_block + scheme->b_reuse_per_block;
    scheme->total_reuse = scheme->total_reuse_per_block * scheme->big_block_count;
}

static void count_block_sets(uint16_t occ[L2_SETS], uint64_t base_addr, int rows,
                             uint64_t row_stride_bytes, uint64_t row_bytes) {
    for (int row = 0; row < rows; row++) {
        uint64_t row_base = base_addr + (uint64_t)row * row_stride_bytes;
        for (uint64_t offset = 0; offset < row_bytes; offset += CACHE_LINE_BYTES) {
            uint64_t addr = row_base + offset;
            uint64_t set = (addr / CACHE_LINE_BYTES) % L2_SETS;
            occ[set]++;
        }
    }
}

static uint16_t max_read_set_occupancy(const matrix_layout_t *layout,
                                       const block_scheme_t *scheme) {
    uint16_t max_occ = 0;

    for (int mb = 0; mb < M; mb += scheme->bm) {
        for (int kb = 0; kb < K; kb += scheme->bk) {
            for (int nb = 0; nb < N; nb += scheme->bn) {
                uint16_t occ[L2_SETS] = {0};
                uint64_t a_base = layout->a_base + (uint64_t)mb * (uint64_t)K + (uint64_t)kb;
                uint64_t b_base = layout->b_base + (uint64_t)kb * (uint64_t)N + (uint64_t)nb;

                count_block_sets(occ, a_base, scheme->bm, (uint64_t)K * sizeof(int8_t),
                                 (uint64_t)scheme->bk * sizeof(int8_t));
                count_block_sets(occ, b_base, scheme->bk, (uint64_t)N * sizeof(int8_t),
                                 (uint64_t)scheme->bn * sizeof(int8_t));

                for (int set = 0; set < L2_SETS; set++) {
                    if (occ[set] > max_occ) {
                        max_occ = occ[set];
                    }
                }
            }
        }
    }

    return max_occ;
}

static uint16_t max_c_set_occupancy(const matrix_layout_t *layout,
                                    const block_scheme_t *scheme) {
    uint16_t max_occ = 0;

    for (int mb = 0; mb < M; mb += scheme->bm) {
        for (int nb = 0; nb < N; nb += scheme->bn) {
            uint16_t occ[L2_SETS] = {0};
            uint64_t c_base = layout->c_base + ((uint64_t)mb * (uint64_t)N + (uint64_t)nb) * sizeof(int32_t);

            count_block_sets(occ, c_base, scheme->bm, (uint64_t)N * sizeof(int32_t),
                             (uint64_t)scheme->bn * sizeof(int32_t));

            for (int set = 0; set < L2_SETS; set++) {
                if (occ[set] > max_occ) {
                    max_occ = occ[set];
                }
            }
        }
    }

    return max_occ;
}

static uint16_t max_combined_set_occupancy(const matrix_layout_t *layout,
                                           const block_scheme_t *scheme) {
    uint16_t max_occ = 0;

    for (int mb = 0; mb < M; mb += scheme->bm) {
        for (int kb = 0; kb < K; kb += scheme->bk) {
            for (int nb = 0; nb < N; nb += scheme->bn) {
                uint16_t occ[L2_SETS] = {0};
                uint64_t a_base = layout->a_base + (uint64_t)mb * (uint64_t)K + (uint64_t)kb;
                uint64_t b_base = layout->b_base + (uint64_t)kb * (uint64_t)N + (uint64_t)nb;
                uint64_t c_base = layout->c_base +
                                  ((uint64_t)mb * (uint64_t)N + (uint64_t)nb) * sizeof(int32_t);

                count_block_sets(occ, a_base, scheme->bm, (uint64_t)K * sizeof(int8_t),
                                 (uint64_t)scheme->bk * sizeof(int8_t));
                count_block_sets(occ, b_base, scheme->bk, (uint64_t)N * sizeof(int8_t),
                                 (uint64_t)scheme->bn * sizeof(int8_t));
                count_block_sets(occ, c_base, scheme->bm, (uint64_t)N * sizeof(int32_t),
                                 (uint64_t)scheme->bn * sizeof(int32_t));

                for (int set = 0; set < L2_SETS; set++) {
                    if (occ[set] > max_occ) {
                        max_occ = occ[set];
                    }
                }
            }
        }
    }

    return max_occ;
}

static int scheme_dominates(const block_scheme_t *lhs, const block_scheme_t *rhs) {
    int no_smaller = lhs->bm >= rhs->bm && lhs->bk >= rhs->bk && lhs->bn >= rhs->bn;
    int strictly_larger = lhs->bm > rhs->bm || lhs->bk > rhs->bk || lhs->bn > rhs->bn;
    return no_smaller && strictly_larger;
}

static int choose_best_scheme(int cache_kb, const matrix_layout_t *layout, block_scheme_t *best) {
    uint64_t cache_bytes = (uint64_t)cache_kb * 1024ULL;
    uint64_t best_score = 0;
    uint64_t best_ws = 0;
    int found = 0;
    size_t candidate_count = 0;
    size_t candidate_cap = 64;
    block_scheme_t *candidates = (block_scheme_t *)malloc(candidate_cap * sizeof(block_scheme_t));

    if (candidates == NULL) {
        fprintf(stderr, "failed to allocate candidate list\n");
        return -1;
    }

    if (M < BLOCK_M || K < BLOCK_K || N < BLOCK_N) {
        fprintf(stderr, "M/K/N must be at least %d/%d/%d\n", BLOCK_M, BLOCK_K, BLOCK_N);
        return -1;
    }
    if (M % BLOCK_M != 0 || K % BLOCK_K != 0 || N % BLOCK_N != 0) {
        fprintf(stderr, "M/K/N must be multiples of %d/%d/%d, got %d/%d/%d\n",
                BLOCK_M, BLOCK_K, BLOCK_N, M, K, N);
        return -1;
    }

    for (int bm = BLOCK_M; bm <= M; bm += BLOCK_M) {
        if (M % bm != 0) {
            continue;
        }
        for (int bk = BLOCK_K; bk <= K; bk += BLOCK_K) {
            if (K % bk != 0) {
                continue;
            }
            for (int bn = BLOCK_N; bn <= N; bn += BLOCK_N) {
                if (N % bn != 0) {
                    continue;
                }

                uint64_t a_bytes = (uint64_t)bm * (uint64_t)bk * sizeof(int8_t);
                uint64_t b_bytes = (uint64_t)bk * (uint64_t)bn * sizeof(int8_t);
                uint64_t c_bytes = (uint64_t)bm * (uint64_t)bn * sizeof(int32_t);
                uint64_t working_set_bytes = a_bytes + b_bytes + c_bytes;
                if (working_set_bytes > cache_bytes) {
                    continue;
                }

                block_scheme_t candidate = {
                    .bm = bm,
                    .bk = bk,
                    .bn = bn,
                    .working_set_bytes = working_set_bytes,
                };
                fill_scheme_stats(&candidate);
                uint16_t read_max_occ = max_read_set_occupancy(layout, &candidate);
                uint16_t c_max_occ = max_c_set_occupancy(layout, &candidate);
                uint16_t combined_max_occ = max_combined_set_occupancy(layout, &candidate);
                if (read_max_occ > L2_WAYS || c_max_occ > L2_WAYS) {
                    printf("reject block %d x %d x %d | ws=%lluB | read_max_set_occ=%u | c_max_set_occ=%u | combined_max_set_occ=%u | ways=%d\n",
                           candidate.bm, candidate.bk, candidate.bn,
                           (unsigned long long)candidate.working_set_bytes,
                           (unsigned int)read_max_occ,
                           (unsigned int)c_max_occ,
                           (unsigned int)combined_max_occ,
                           L2_WAYS);
                    continue;
                }

                if (candidate_count == candidate_cap) {
                    candidate_cap *= 2;
                    block_scheme_t *next_candidates = (block_scheme_t *)realloc(
                        candidates, candidate_cap * sizeof(block_scheme_t));
                    if (next_candidates == NULL) {
                        free(candidates);
                        fprintf(stderr, "failed to grow candidate list\n");
                        return -1;
                    }
                    candidates = next_candidates;
                }
                candidates[candidate_count++] = candidate;
            }
        }
    }

    for (size_t i = 0; i < candidate_count; i++) {
        int dominated = 0;
        for (size_t j = 0; j < candidate_count; j++) {
            if (i == j) {
                continue;
            }
            if (scheme_dominates(&candidates[j], &candidates[i])) {
                dominated = 1;
                break;
            }
        }

        if (dominated) {
            // printf("skip dominated candidate %d x %d x %d\n",
            //        candidates[i].bm, candidates[i].bk, candidates[i].bn);
            continue;
        }

        uint64_t score = candidates[i].total_reuse;
        uint16_t read_max_occ = max_read_set_occupancy(layout, &candidates[i]);
        uint16_t c_max_occ = max_c_set_occupancy(layout, &candidates[i]);
        uint16_t combined_max_occ = max_combined_set_occupancy(layout, &candidates[i]);
        printf("candidate block %d x %d x %d | ws=%lluB | read_max_set_occ=%u | c_max_set_occ=%u | combined_max_set_occ=%u | blocks=%llu | kernels/block=%llu | A_reuse/block=%llu | B_reuse/block=%llu | total_reuse/block=%llu | total_reuse=%llu\n",
               candidates[i].bm, candidates[i].bk, candidates[i].bn,
               (unsigned long long)candidates[i].working_set_bytes,
               (unsigned int)read_max_occ,
               (unsigned int)c_max_occ,
               (unsigned int)combined_max_occ,
               (unsigned long long)candidates[i].big_block_count,
               (unsigned long long)candidates[i].kernels_per_block,
               (unsigned long long)candidates[i].a_reuse_per_block,
               (unsigned long long)candidates[i].b_reuse_per_block,
               (unsigned long long)candidates[i].total_reuse_per_block,
               (unsigned long long)candidates[i].total_reuse);

        if (!found || score > best_score || (score == best_score && candidates[i].working_set_bytes > best_ws)) {
            found = 1;
            best_score = score;
            best_ws = candidates[i].working_set_bytes;
            *best = candidates[i];
        }
    }

    free(candidates);

    return found ? 0 : -1;
}

static void emit_blocked_trace(const matrix_layout_t *layout, const block_scheme_t *scheme) {
    for (int mb = 0; mb < M; mb += scheme->bm) {
        for (int nb = 0; nb < N; nb += scheme->bn) {
#if MATRIX_REREAD_B_PER_STRIP
            for (int cm = 0; cm < scheme->bm; cm += C_STRIP_M) {
                int rows = C_STRIP_M;
                if (cm + rows > scheme->bm) {
                    rows = scheme->bm - cm;
                }

                for (int kb = 0; kb < K; kb += scheme->bk) {
                    for (int cn = 0; cn < scheme->bn; cn += BLOCK_N) {
                        uint64_t a_base = layout->a_base +
                                          ((uint64_t)mb + (uint64_t)cm) * (uint64_t)K +
                                          (uint64_t)kb;
                        uint64_t b_base = layout->b_base + (uint64_t)kb * (uint64_t)N +
                                          (uint64_t)nb + (uint64_t)cn;

                        /*
                         * Emit the A/B reads consumed by each microkernel. Repeated
                         * reads inside the chosen big block should hit in L2, which is
                         * the behavior Matrix Get is meant to accelerate.
                         */
                        trace_matrix_block('r', a_base, rows, (uint64_t)K * sizeof(int8_t),
                                           (uint64_t)scheme->bk * sizeof(int8_t), 'a');
                        trace_matrix_block('r', b_base, scheme->bk, (uint64_t)N * sizeof(int8_t),
                                           (uint64_t)BLOCK_N * sizeof(int8_t), 'b');
                    }
                }

                uint64_t c_base = layout->c_base +
                                  (((uint64_t)mb + (uint64_t)cm) * (uint64_t)N + (uint64_t)nb) *
                                  sizeof(int32_t);
                trace_matrix_block('w', c_base, rows, (uint64_t)N * sizeof(int32_t),
                                   (uint64_t)scheme->bn * sizeof(int32_t), 'c');
            }
#else
#if MATRIX_EMIT_REUSE_READS
            for (int kb = 0; kb < K; kb += scheme->bk) {
                for (int cm = 0; cm < scheme->bm; cm += BLOCK_M) {
                    for (int cn = 0; cn < scheme->bn; cn += BLOCK_N) {
                        uint64_t a_base = layout->a_base +
                                          ((uint64_t)mb + (uint64_t)cm) * (uint64_t)K +
                                          (uint64_t)kb;
                        uint64_t b_base = layout->b_base + (uint64_t)kb * (uint64_t)N +
                                          (uint64_t)nb + (uint64_t)cn;

                        trace_matrix_block('r', a_base, BLOCK_M, (uint64_t)K * sizeof(int8_t),
                                           (uint64_t)scheme->bk * sizeof(int8_t), 'a');
                        trace_matrix_block('r', b_base, scheme->bk, (uint64_t)N * sizeof(int8_t),
                                           (uint64_t)BLOCK_N * sizeof(int8_t), 'b');
                    }
                }
            }
#else
            for (int kb = 0; kb < K; kb += scheme->bk) {
                uint64_t a_base = layout->a_base + (uint64_t)mb * (uint64_t)K + (uint64_t)kb;
                uint64_t b_base = layout->b_base + (uint64_t)kb * (uint64_t)N + (uint64_t)nb;

                trace_matrix_block('r', a_base, scheme->bm, (uint64_t)K * sizeof(int8_t),
                                   (uint64_t)scheme->bk * sizeof(int8_t), 'a');
                trace_matrix_block('r', b_base, scheme->bk, (uint64_t)N * sizeof(int8_t),
                                   (uint64_t)scheme->bn * sizeof(int8_t), 'b');
            }
#endif

            for (int cm = 0; cm < scheme->bm; cm += C_STRIP_M) {
                int rows = C_STRIP_M;
                if (cm + rows > scheme->bm) {
                    rows = scheme->bm - cm;
                }

                uint64_t c_base = layout->c_base +
                                  (((uint64_t)mb + (uint64_t)cm) * (uint64_t)N + (uint64_t)nb) *
                                  sizeof(int32_t);
                trace_matrix_block('w', c_base, rows, (uint64_t)N * sizeof(int32_t),
                                   (uint64_t)scheme->bn * sizeof(int32_t), 'c');
            }
#endif
        }
    }
}

int main(int argc, char **argv) {
    matrix_layout_t layout = build_layout();
    block_scheme_t scheme = {0};
    int cache_kb = 256;

    if (argc > 1) {
        cache_kb = atoi(argv[1]);
        if (cache_kb <= 0) {
            fprintf(stderr, "invalid cache size: %s\n", argv[1]);
            return 1;
        }
    }

    if (choose_best_scheme(cache_kb, &layout, &scheme) != 0) {
        fprintf(stderr, "failed to choose a block scheme\n");
        return 1;
    }

    printf("# M=%d K=%d N=%d\n", M, K, N);
    printf("# cache=%dKB\n", cache_kb);
    printf("# best_block=%d x %d x %d | ws=%lluB | read_max_set_occ=%u | c_max_set_occ=%u | combined_max_set_occ=%u | blocks=%llu | reuse/block=%llu | total_reuse=%llu\n",
           scheme.bm, scheme.bk, scheme.bn,
           (unsigned long long)scheme.working_set_bytes,
           (unsigned int)max_read_set_occupancy(&layout, &scheme),
           (unsigned int)max_c_set_occupancy(&layout, &scheme),
           (unsigned int)max_combined_set_occupancy(&layout, &scheme),
           (unsigned long long)scheme.big_block_count,
           (unsigned long long)scheme.total_reuse_per_block,
           (unsigned long long)scheme.total_reuse);
    printf("# A_BASE=%llu B_BASE=%llu C_BASE=%llu\n",
           (unsigned long long)layout.a_base,
           (unsigned long long)layout.b_base,
           (unsigned long long)layout.c_base);

    emit_blocked_trace(&layout, &scheme);

    printf("# trace_summary checksum=%llu\n", (unsigned long long)scheme.total_reuse);
    return 0;
}
