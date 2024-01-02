#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

using namespace std;

// C = A * B
int main(int argc, char const *argv[])
{
    int M = 4;  // 矩阵A的行数，矩阵C的行数
    int N = 4;  // 矩阵A的列数，矩阵B的行数
    int K = 4;  // 矩阵B的列数，矩阵C的列数

    // init data buffer
    float *host_a;
    float *host_b;
    float *host_c1;  // 用于接收stream1的result
    float *host_c2;  // 用于接收stream2的result
    cudaHostAlloc((void **)&host_a, M*N*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, N*K*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c1, M*K*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c2, M*K*sizeof(float), cudaHostAllocDefault);

    // create data
    for (size_t i = 0; i < M * N; i++)
    {
        host_a[i] = i;
    }

    for (size_t i = 0; i < N * K; i++)
    {
        host_b[i] = i;
    }

    // init device for stream1
    float *dev_a1;
    float *dev_b1;
    float *dev_c1;

    // init device for stream2
    float *dev_a2;
    float *dev_b2;
    float *dev_c2;

    cudaMalloc((void **)&dev_a1, M*N*sizeof(float));
    cudaMalloc((void **)&dev_b1, N*K*sizeof(float));
    cudaMalloc((void **)&dev_c1, M*K*sizeof(float));
    cudaMalloc((void **)&dev_a2, M*N*sizeof(float));
    cudaMalloc((void **)&dev_b2, N*K*sizeof(float));
    cudaMalloc((void **)&dev_c2, M*K*sizeof(float));

    // create streams
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // create handles
    cublasHandle_t handle1;
    cublasHandle_t handle2;
    cublasCreate(&handle1);
    cublasCreate(&handle2);

    // 为handle设置stream
    cublasSetStream(handle1, stream1);
    cublasSetStream(handle2, stream2);

    // 将数据拷入device中
    cudaMemcpyAsync(dev_a1, host_a, M*N*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_b1, host_b, N*K*sizeof(float), cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(dev_a2, host_a, M*N*sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_b2, host_b, N*K*sizeof(float), cudaMemcpyHostToDevice, stream2);

    float alpha = 1; 
    float beta = 0;

    // 调用cublas中的gemm函数
    cublasSgemm(handle1, 
                    CUBLAS_OP_N,   // 矩阵A是否转置
                    CUBLAS_OP_N,   // 矩阵B是否转置
                    M, 
                    N, 
                    K, 
                    &alpha,   // 乘积的值
                    dev_a1, 
                    M,        // 数据存储的是连续的地址，M的作用是A每隔几个数据作为一行
                    dev_b1, 
                    N, 
                    &beta, 
                    dev_c1, 
                    M);

    cublasSgemm(handle2, 
                    CUBLAS_OP_N,   // 矩阵A是否转置
                    CUBLAS_OP_N,   // 矩阵B是否转置
                    M, 
                    N, 
                    K, 
                    &alpha,   // 乘积的值
                    dev_a2, 
                    M,        // 数据存储的是连续的地址，M的作用是A每隔几个数据作为一行
                    dev_b2, 
                    N, 
                    &beta, 
                    dev_c2, 
                    M);

    // 将数据拷贝回host
    cudaMemcpyAsync(host_c1, dev_c1, M*K*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(host_c2, dev_c2, M*K*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 打印计算结果
    cout << "Result:" << endl;
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            cout << host_c1[j+i*M] << " ";
            if ((j+1+i*M) % M == 0)
            {
                cout << endl;
            }
        }
    }
    

    // 资源释放
    cudaFree(dev_a1);
    cudaFree(dev_a2);
    cudaFree(dev_b1);
    cudaFree(dev_b2);
    cudaFree(dev_c1);
    cudaFree(dev_c2);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c1);
    cudaFreeHost(host_c2);

    cublasDestroy(handle1);
    cublasDestroy(handle2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);


    
    return 0;
}

// nvcc cublas_base.cu -o cublas_base.out -lcublas