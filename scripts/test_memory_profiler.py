import numpy as np
import psutil
from memory_profiler import profile

@profile
def measure_memory_usage():
    # 创建一个大型的NumPy数组
    arr = np.ones((1000, 1000))

if __name__ == "__main__":
    # 测试内存使用
    measure_memory_usage()
