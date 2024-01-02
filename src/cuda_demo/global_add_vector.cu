#include <stdio.h>
// 导入cuda所需的运行库
#include <cuda_runtime.h>
// A+B=C


__global__ void vectorAdd(const float* A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }

}


int main(int argc, char const *argv[])
{
    // A/B/C 元素总数
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("Vector addition of %d elements.\n", numElements);

    // 在CPU端给ABC三个向量申请存储空间
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // init
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    // 在GPU当中给ABC三个向量申请存储空间
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 把数据AB从CPU内存当中复制到GPU显存当中
    printf("Copy input data from the host memory to device memory\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 执行GPU kernel函数
    int threadsPerBlock = 256;  // 一个block中的线程数量
    int blockPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;  // block数量
    vectorAdd<<<blockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements; ++i)
    {
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at elemet %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("test passed!\n");

    return 0;
}
