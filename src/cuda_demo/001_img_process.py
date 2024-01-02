from numba import cuda
import os
import numba
import cv2
import numpy as np
import math
import time

@numba.jit(nopython=True)
def main_cpu(img, dst):
    h, w, c = img.shape
    for y in range(h):
        for x in range(w):
            for i in range(c):
                color = img[y, x, i] * 2 + 20
                if color < 0:
                    dst[y, x, i] = 0
                elif color > 255:
                    dst[y, x, i] = 255
                else:
                    dst[y, x, i] = color


@cuda.jit
def main_gpu(img, dst):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    for i in range(3):
        color = img[tx, ty, i] * 2 + 20
        if color < 0:
            dst[tx, ty, i] = 0
        elif color > 255:
            dst[tx, ty, i] = 255
        else:
            dst[tx, ty, i] = color


if __name__ == "__main__":

    TPB = 16

    im_path = "data/test.jpg"
    im_name = os.path.basename(im_path)

    img = cv2.imread(im_path)
    rows, cols, channels = img.shape
    dst = np.zeros_like(img, dtype=np.uint8)
    ####### CPU #######
    start_time = time.time()
    main_cpu(img, dst)
    cv2.imwrite("cpu_{}".format(im_name), dst)
    print("cpu cost: {} s".format(time.time() - start_time))

    ##### memcopy #####
    start_time = time.time()
    img_gloal_mem = cuda.to_device(img)
    dst_global_mem = cuda.device_array((img.shape))
    print("memory copy cost: {} s".format(time.time() - start_time))

    ##### grid #####
    start_time = time.time()
    threads_per_block = (TPB, TPB)
    grid_x_dim = int(math.ceil(rows / threads_per_block[0]))
    grid_y_dim = int(math.ceil(cols / threads_per_block[1]))
    blacks_per_grid = (grid_x_dim, grid_y_dim)
    print("create grid cost: {} s".format(time.time() - start_time))

    ####### GPU #######
    start_time = time.time()
    cuda.synchronize()
    main_gpu[blacks_per_grid, threads_per_block](img_gloal_mem, dst_global_mem)
    cuda.synchronize()
    dst_img = dst_global_mem.copy_to_host()
    cv2.imwrite("gpu_{}".format(im_name), dst_img)
    print("gpu cost: {} s".format(time.time() - start_time))
