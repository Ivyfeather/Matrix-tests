#include "../../isa/utils.h"
#include "../ops/matmul.h"
#include "../ops/tensor.h"
#include <riscv_matrix.h>

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#define MAX_ELEMENT_LENGTH 128

float* read_matrix_file(const char* file_name, int rows, int cols) {
    FILE *file = fopen(file_name, "r");
    printf("open filename: %s\n", file_name);
    if (!file) {
        printf("Errorno: %d\n", errno);
        perror("Unable to open file");
        return NULL;
    }
    // 分配一维数组的内存
    int size = rows * cols;
    float *matrix = malloc(size * sizeof(float));
    if (!matrix) {
        perror("Failed to allocate memory for matrix");
        fclose(file);
        return NULL;
    }

    size_t bytesRead = fread(matrix, sizeof(float), size, file);
    if (bytesRead != size) {
        perror("Unable to read matrix from file");
        free(matrix);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return matrix;
}

void trap(int trap_code) {
    asm volatile (
        "mv a0, %0\n"
        ".word 0x5006b\n"
        :
        : "r"(trap_code)
        : "a0"
    );
}


int main() {
  const int M = 4096;
  const int K = 4096;
  const int N = 1024;
  
  float *src1 = read_matrix_file("matrixA.bin", M, K);
  float *src2 = read_matrix_file("matrixB.bin", K, N);
  float *dest = read_matrix_file("matrixC.bin", M, N);
  float *answ = read_matrix_file("resultC.bin", M, N);
  
  create_tensor4d(matmul_src1, (void *)src1, 1, M, K, 1);
  create_tensor4d(matmul_src2, (void *)src2, 1, K, N, 1);
  create_tensor4d(matmul_dest, (void *)dest, 1, M, N, 1);
  create_tensor4d(matmul_answ, (void *)answ, 1, M, N, 1);
  matmul_2x2(&matmul_dest, &matmul_src1, &matmul_src2);
  float *dst_ptr = (float *)matmul_dest.data;
  float *ans_ptr = (float *)matmul_answ.data;
  EXCEPT_FP32_ARRAY_LAX_EQ(ans_ptr, dst_ptr, M * N,
                           "MATMUL [4096, 4096] @ [4096, 1024]");

  free(src1);
  free(src2);
  free(dest);
  free(answ);
  
  trap(0);
  return 0;
}