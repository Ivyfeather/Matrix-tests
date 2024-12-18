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
    float *matrix = malloc(rows * cols * sizeof(float));
    if (!matrix) {
        perror("Failed to allocate memory for matrix");
        fclose(file);
        return NULL;
    }

    char element[MAX_ELEMENT_LENGTH];
    int c;
    int i = 0;
    int elementIndex = 0;
    char *end;

    while ((c = fgetc(file)) != EOF) {
        if (c == ',') {
            // 遇到逗号，表示一个元素结束
            element[i] = '\0';
            matrix[elementIndex++] = strtof(element, &end);
            if (end == element) {
                printf("Error: can not translate str to float.\n");
                free(matrix);
                fclose(file);
                return NULL;
            }
            i = 0;
        } else if (c == '\n' || c == '\r') {
            // 遇到换行符，表示一行结束
            if (i > 0) {
                element[i] = '\0';
                matrix[elementIndex++] = strtof(element, &end);
                if (end == element) {
                    printf("Error: can not translate str to float.\n");
                    free(matrix);
                    fclose(file);
                    return NULL;
                }
                i = 0;
            }
        } else {
            // 其他字符，认为是数字的一部分
            element[i++] = (char)c;
            if (i >= sizeof(element)) {
                fprintf(stderr, "Error: number too long in CSV file.\n");
                free(matrix);
                fclose(file);
                return NULL;
            }
        }
    }

    // 如果最后一行没有以换行符结束，需要处理最后一个元素
    if (i > 0) {
        element[i] = '\0';
        matrix[elementIndex++] = strtof(element, &end);
        if (end == element) {
            printf("Error: can not translate str to float.\n");
            free(matrix);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return matrix;
}


int main() {
  const int M = 32;
  const int K = 32;
  const int N = 32;
  
  float *src1 = read_matrix_file("/root/matrix-tests/matrixA.csv", M, K);
  float *src2 = read_matrix_file("matrixB.csv", K, N);
  float *dest = read_matrix_file("matrixC.csv", M, N);
  float *answ = read_matrix_file("resultC.csv", M, N);
  
  create_tensor4d(matmul_src1, (void *)src1, 1, M, K, 1);
  create_tensor4d(matmul_src2, (void *)src2, 1, K, N, 1);
  create_tensor4d(matmul_dest, (void *)dest, 1, M, N, 1);
  create_tensor4d(matmul_answ, (void *)answ, 1, M, N, 1);
  matmul_2x2(&matmul_dest, &matmul_src1, &matmul_src2);
  float *dst_ptr = (float *)matmul_dest.data;
  float *ans_ptr = (float *)matmul_answ.data;
  EXCEPT_FP32_ARRAY_LAX_EQ(ans_ptr, dst_ptr, M * N,
                           "MATMUL [32, 32] @ [32, 32]");
  
  free(src1);
  free(src2);
  free(dest);
  free(answ);
  return 0;
}