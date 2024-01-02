from numba import cuda, float32
import numba
import numpy as np
import math
import time

TPB = 16


@numba.jit(nopython=True)
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0.0
            for k in range(A.shape[1]):  # or B.shape[0]
                tmp += A[x, k] * B[k, y]
            C[x, y] = tmp


@cuda.jit
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@cuda.jit
def matmul_shared_mem_gpu(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.0
    for i in range(int(A.shape[1] / TPB)):
        sA[tx, ty] = A[x, ty+i*TPB]
        sB[tx, ty] = B[tx+i*TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        # 一定要加上同步，否则CPU不会等你执行完
        cuda.syncthreads()

    C[x, y] = tmp


A = np.full((TPB*500, TPB*1000), 3, dtype=np.float)
B = np.full((TPB*1000, TPB*200), 4, dtype=np.float)
C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float)

####### CPU #######
start_time = time.time()
# matmul_cpu(A, B, C)
print("cpu cost: {} s".format(time.time() - start_time))


##### memcopy #####
start_time = time.time()
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
C_shared_mem = cuda.device_array((A.shape[0], B.shape[1]))
print("memory copy cost: {} s".format(time.time() - start_time))


##### grid #####
start_time = time.time()
threads_per_block = (TPB, TPB)
grid_x_dim = int(math.ceil(A.shape[0] / threads_per_block[0]))
grid_y_dim = int(math.ceil(B.shape[1] / threads_per_block[1]))
blacks_per_grid = (grid_x_dim, grid_y_dim)
print("create grid cost: {} s".format(time.time() - start_time))

####### GPU #######
start_time = time.time()
matmul_gpu[blacks_per_grid, threads_per_block](
    A_global_mem, B_global_mem, C_global_mem)
cuda.synchronize()
print("gpu cost: {} s".format(time.time() - start_time))


####### GPU shared mem #######
start_time = time.time()
matmul_shared_mem_gpu[blacks_per_grid, threads_per_block](
    A_global_mem, B_global_mem, C_shared_mem)
cuda.synchronize()
print("gpu cost: {} s".format(time.time() - start_time))
