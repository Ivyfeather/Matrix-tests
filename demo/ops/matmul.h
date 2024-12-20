#ifndef _MATMUL_H_
#define _MATMUL_H_

#include "tensor.h"
#include <riscv_matrix.h>

void matmul(Tensor *dest, Tensor *src1, Tensor *src2) {
  float *dest_ptr = (float *)dest->data;
  float *src1_ptr = (float *)src1->data;
  float *src2_ptr = (float *)src2->data;
  const int M = src1->shape[1];
  const int K = src1->shape[2];
  const int N = src2->shape[2];
  int tile_m, tile_k, tile_n;
  msettype(E32, M1, BA);

  for (int m = 0; m < M; m += tile_m) {
    tile_m = msettilem(M - m);

    for (int n = 0; n < N; n += tile_n) {
      tile_n = msettilen(N - n);
      mfloat32m1_t out = mlce32_m1(dest_ptr + m * N + n, N * sizeof(float));
      
      for (int k = 0; k < K; k += tile_k) {
        tile_k = msettilek(K - k);
        mfloat32m1_t tr0 = mlae32_m1(src1_ptr + m * K + k, K * sizeof(float));
        mfloat32m1_t tr1 = mlbe32_m1(src2_ptr + k * N + n, N * sizeof(float));
        out = mfma_mm(out, tr0, tr1);
      }
      msce32_m(out, dest_ptr + m * N + n, N * sizeof(float));
    }
  }
}

void matmul_2x2(Tensor *dest, Tensor *src1, Tensor *src2) {
  float *dest_ptr = (float *)dest->data;
  float *src1_ptr = (float *)src1->data;
  float *src2_ptr = (float *)src2->data;
  const int M = src1->shape[1];
  const int K = src1->shape[2];
  const int N = src2->shape[2];
  int tile_m, tile_k, tile_n;
  msettype(E32, M1, BA);

  tile_m = msettilem(M);
  int m = 0;
  // 2row(a) x n(b)
  for (; m + 2 * tile_m - 1 < M; m += 2 * tile_m) {
    tile_m = msettilem(M-m);
    
    tile_n = msettilen(N);
    int n = 0;
    // 2row(a) x 2col(b)
    for (; n + 2 * tile_n - 1 < N; n += 2 * tile_n) {
      tile_n = msettilen(N - n);
      mfloat32m1_t out_4 = mlce32_m1(dest_ptr + m * N + n,                       N * sizeof(float));
      mfloat32m1_t out_5 = mlce32_m1(dest_ptr + m * N + (n + tile_n),            N * sizeof(float));
      mfloat32m1_t out_6 = mlce32_m1(dest_ptr + (m + tile_m) * N + n,            N * sizeof(float));
      mfloat32m1_t out_7 = mlce32_m1(dest_ptr + (m + tile_m) * N + (n + tile_n), N * sizeof(float));

      for (int k = 0; k < K; k += tile_k) {
        tile_k = msettilek(K - k);
        mfloat32m1_t tr0_a = mlae32_m1(src1_ptr + m * K + k,            K * sizeof(float));
        mfloat32m1_t tr1_a = mlae32_m1(src1_ptr + (m + tile_m) * K + k, K * sizeof(float));
        mfloat32m1_t tr2_b = mlbe32_m1(src2_ptr + k * N + n,            N * sizeof(float));
        mfloat32m1_t tr3_b = mlbe32_m1(src2_ptr + k * N + (n + tile_n), N * sizeof(float));
        out_4 = mfma_mm(out_4, tr0_a, tr2_b);
        out_5 = mfma_mm(out_5, tr0_a, tr3_b);
        out_6 = mfma_mm(out_6, tr1_a, tr2_b);
        out_7 = mfma_mm(out_7, tr1_a, tr3_b);
      }
      msce32_m(out_4, dest_ptr + m * N + n,                       N * sizeof(float));
      msce32_m(out_5, dest_ptr + m * N + (n + tile_n),            N * sizeof(float));
      msce32_m(out_6, dest_ptr + (m + tile_m) * N + n,            N * sizeof(float));
      msce32_m(out_7, dest_ptr + (m + tile_m) * N + (n + tile_n), N * sizeof(float));
    }

    tile_n = msettilen(N-n);
    // 2row(a) x rest(b)
    for (; n < N; n += tile_n) {
      tile_n = msettilen(N - n);
      mfloat32m1_t out_4 = mlce32_m1(dest_ptr + m * N + n,            N * sizeof(float));
      mfloat32m1_t out_5 = mlce32_m1(dest_ptr + (m + tile_m) * N + n, N * sizeof(float));

      for (int k = 0; k < K; k += tile_k) {
        tile_k = msettilek(K - k);
        mfloat32m1_t tr0_a = mlae32_m1(src1_ptr + m * K + k,            K * sizeof(float));
        mfloat32m1_t tr1_a = mlae32_m1(src1_ptr + (m + tile_m) * K + k, K * sizeof(float));
        mfloat32m1_t tr2_b = mlbe32_m1(src2_ptr + k * N + n,            N * sizeof(float));
        out_4 = mfma_mm(out_4, tr0_a, tr2_b);
        out_5 = mfma_mm(out_5, tr1_a, tr2_b);
      }
      msce32_m(out_4, dest_ptr + m * N + n,            N * sizeof(float));
      msce32_m(out_5, dest_ptr + (m + tile_m) * N + n, N * sizeof(float));
    }
  }

  tile_m = msettilem(M-m);
  // rest(a) x n(b)
  for (; m < M; m += tile_m) {
    tile_m = msettilem(M-m);
    
    tile_n = msettilen(N);
    int n = 0;
    // rest(a) x 2col(b)
    for (; n + 2 * tile_n - 1 < N; n += 2 * tile_n) {
      tile_n = msettilen(N - n);
      mfloat32m1_t out_4 = mlce32_m1(dest_ptr + m * N + n,            N * sizeof(float));
      mfloat32m1_t out_5 = mlce32_m1(dest_ptr + m * N + (n + tile_n), N * sizeof(float));

      for (int k = 0; k < K; k += tile_k) {
        tile_k = msettilek(K - k);
        mfloat32m1_t tr0_a = mlae32_m1(src1_ptr + m * K + k,            K * sizeof(float));
        mfloat32m1_t tr1_b = mlbe32_m1(src2_ptr + k * N + n,            N * sizeof(float));
        mfloat32m1_t tr2_b = mlbe32_m1(src2_ptr + k * N + (n + tile_n), N * sizeof(float));
        out_4 = mfma_mm(out_4, tr0_a, tr1_b);
        out_5 = mfma_mm(out_5, tr0_a, tr2_b);
      }
      msce32_m(out_4, dest_ptr + m * N + n,                       N * sizeof(float));
      msce32_m(out_5, dest_ptr + m * N + (n + tile_n),            N * sizeof(float));
    }

    tile_n = msettilen(N-n);
    // rest(a) x rest(b)
    for (; n < N; n += tile_n) {
      tile_n = msettilen(N - n);
      mfloat32m1_t out = mlce32_m1(dest_ptr + m * N + n, N * sizeof(float));

      for (int k = 0; k < K; k += tile_k) {
        tile_k = msettilek(K - k);
        mfloat32m1_t tr_a = mlae32_m1(src1_ptr + m * K + k, K * sizeof(float));
        mfloat32m1_t tr_b = mlbe32_m1(src2_ptr + k * N + n, N * sizeof(float));
        out = mfma_mm(out, tr_a, tr_b);
      }
      msce32_m(out, dest_ptr + m * N + n, N * sizeof(float));
    }
  }

}

#endif