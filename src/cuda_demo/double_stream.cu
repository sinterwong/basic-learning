#include <stdio.h>

#define N (1024 * 1024)
#define FULL (N * 20)

__global__ void kernel(int *a, int *b, int *c)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
    {
        c[idx] = (a[idx] + b[idx]) / 2;
    }
}

int main(int argc, char const *argv[])
{

    // 检查GPU属性是否支持Stream
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap)
    {
        printf("Device will not support overlap!\n");
        return 0;
    }

    // init 计时 event & create 计时器
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsed;

    // init stream & create
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 声明buffer & 锁页内存分配 & cuda内存分配
    int *host_a;
    int *host_b;
    int *host_c;
    int *dev_a1;
    int *dev_b1;
    int *dev_c1;
    int *dev_a2;
    int *dev_b2;
    int *dev_c2;
    cudaHostAlloc((void **)&host_a, FULL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, FULL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c, FULL * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_a1, N * sizeof(int));
    cudaMalloc((void **)&dev_b1, N * sizeof(int));
    cudaMalloc((void **)&dev_c1, N * sizeof(int));
    cudaMalloc((void **)&dev_a2, N * sizeof(int));
    cudaMalloc((void **)&dev_b2, N * sizeof(int));
    cudaMalloc((void **)&dev_c2, N * sizeof(int));

    // A和B赋值
    for (size_t i = 0; i < FULL; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    // 开始计时
    cudaEventRecord(start, 0);

    // 多流异步计算
    for (size_t i = 0; i < FULL; i+=(N*2))
    {
        // 将锁页内存拷贝到Device
        cudaMemcpyAsync(dev_a1, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_a2, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream2);

        cudaMemcpyAsync(dev_b1, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b2, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream2);

        kernel<<<N/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
        kernel<<<N/256, 256, 0, stream2>>>(dev_a2, dev_b2, dev_c2);

        // 将计算结果拷贝回host
        cudaMemcpyAsync(host_c+i, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(host_c+i+N, dev_c2, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    }
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("cost time: %3.3f ms\n", elapsed);

    // relased resources 
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(dev_a1);
    cudaFree(dev_a2);
    cudaFree(dev_b1);
    cudaFree(dev_b2);
    cudaFree(dev_c1);
    cudaFree(dev_c2);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);

    return 0;
}
