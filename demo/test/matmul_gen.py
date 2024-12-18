import torch
import numpy as np

def generate_random_matrix(shape, device='cpu'):
    """生成随机矩阵并移动到指定设备（CPU/GPU）。"""
    matrix = torch.randn(shape)
    return matrix.to(device)


if __name__ == "__main__":
    shape1 = (32, 32)
    shape2 = (32, 32)
    shape3 = (32, 32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成随机矩阵(默认数据类型为float32)
    matrixA = generate_random_matrix(shape1, device)
    matrixB = generate_random_matrix(shape2, device)
    matrixC = generate_random_matrix(shape3, device)
    
    # 计算矩阵乘结果
    resultC = torch.add(torch.matmul(matrixA, matrixB), matrixC)

    # 将4个矩阵以csv文件格式保存
    np.savetxt('matrixA.csv', matrixA.numpy(), delimiter=',') # delimiter指定了分隔符，默认为空格，这里设置为逗号
    np.savetxt('matrixB.csv', matrixB.numpy(), delimiter=',')
    np.savetxt('matrixC.csv', matrixB.numpy(), delimiter=',')
    np.savetxt('resultC.csv', resultC.numpy(), delimiter=',')