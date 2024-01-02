#include <stdio.h>

// (A+B)/2=C
#define N (4096 * 4096) // 每个stream执行数据的大小
#define FULL (N * 20)   // 全部数据的大小

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
    // 检查设备属性是否支持Stream
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap)
    {
        printf("Device will not support overlap!\n");
        return 0;
    }

    // init 计时器 event
    cudaEvent_t start, end;
    float elapsedTime;

    // create 计时器
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // 声明并创建Stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 声明Buffer指针
    int *host_a;
    int *host_b;
    int *host_c;

    int *dev_a;
    int *dev_b;
    int *dev_c;

    // 锁页内存分配
    cudaHostAlloc((void **)&host_a, FULL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, FULL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c, FULL * sizeof(int), cudaHostAllocDefault);

    // Device 内存分配
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));
    // cudaHostAlloc((void **)&dev_a, N * sizeof(int), cudaHostAllocDefault);
    // cudaHostAlloc((void **)&dev_b, N * sizeof(int), cudaHostAllocDefault);
    // cudaHostAlloc((void **)&dev_c, N * sizeof(int), cudaHostAllocDefault);

    // 为A和B赋值
    for (int i = 0; i < FULL; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    // 运作计时器
    cudaEventRecord(start, 0);

    // 分流异步计算
    for (int i = 0; i < FULL; i+=N)
    {
        // 将锁页内存上的数据拷贝到Device上
        cudaMemcpyAsync(dev_a, host_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, host_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream);
        
        // <<<gridDim, blockDim, 使用shared_mem大小, stream>>>
        kernel<<<N/256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

        // 将计算结果copy到Host上
        cudaMemcpyAsync(host_c+i, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);

    // 耗时计算
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("cost time: %3.3f ms\n", elapsedTime);
    
    // 资源释放
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);    
    cudaStreamDestroy(stream);


    return 0;
}
