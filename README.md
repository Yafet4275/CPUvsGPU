# Performance Comparison Between GPU and CPU for Matrix Multiplication

## Overview

This Jupyter Notebook compares the performance of matrix multiplication on a GPU versus a CPU. The comparison includes timing the matrix multiplication operations and checking GPU and CPU utilizations during these operations. The GPU used for this experiment is the **NVIDIA GeForce RTX 3060**.

## Requirements

To run this notebook, you need:

- **PyTorch** with GPU support (CUDA) installed. You can install it via pip:
  ```bash
  pip install torch
  pip install psutil


GPU Details
GPU Model: NVIDIA GeForce RTX 3060
Compute Capability: sm_86 (Architecture: Ampere)
Number of Streaming Multiprocessors (SMs): 28
Number of CUDA Cores: 3584
Total Memory: 11938 MB

gpu_properties: _CudaDeviceProperties(name='NVIDIA GeForce RTX 3060', major=8, minor=6, total_memory=11938MB, multi_processor_count=28)
gpu_name: NVIDIA GeForce RTX 3060
sm_count: 28

Estimated CUDA Cores
The number of CUDA cores is calculated based on the architecture and the number of streaming multiprocessors:
Number of CUDA cores: 3584

Code Explanation
1. Check CUDA Availability and Get Device
\`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
This section checks if a GPU with CUDA support is available and selects it. If not, it defaults to the CPU.

2. Get GPU Properties
gpu_properties = torch.cuda.get_device_properties(device)
gpu_name = gpu_properties.name
sm_count = gpu_properties.multi_processor_count
print('gpu_properties: ', gpu_properties)
print('gpu_name: ', gpu_name)
print('sm_count: ', sm_count)
Here, we retrieve and print the GPU properties, including its name and the number of streaming multiprocessors.

3. Estimate Number of CUDA Cores
cores_per_sm = {
    "sm_70": 64,  # V100
    "sm_75": 64,  # T4, 2080Ti
    "sm_80": 64,  # A100
    "sm_86": 128,  # 3090, 3080, A40, A30
    "sm_87": 128,  # A10, A16
    "sm_89": 128,  # RTX 3060
    # Add more architectures if needed
}
arch = f"sm_{gpu_properties.major}{gpu_properties.minor}"
num_cores = sm_count * cores_per_sm.get(arch, 64)  # Default to 64 if architecture is unknown
print(f"Using GPU: {gpu_name}")
print(f"Number of streaming multiprocessors: {sm_count}")
print(f"Number of CUDA cores: {num_cores}")
This section estimates the number of CUDA cores based on the GPU's architecture and number of SMs.

4. Matrix Multiplication on GPU
matrix_size = 10000

a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

start_time = time.time()
result = torch.matmul(a, b)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for matrix multiplication: {elapsed_time} seconds")

gpu_utilization = torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory * 100
print(f"GPU utilization: {gpu_utilization}%")
This section performs matrix multiplication on the GPU and measures the time taken. It also checks GPU memory utilization.

5. Matrix Multiplication on CPU
device = torch.device("cpu")

a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

start_time = time.time()
result = torch.matmul(a, b)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for matrix multiplication on CPU: {elapsed_time} seconds")

import psutil
cpu_usage = psutil.cpu_percent(interval=1)
print(f"CPU utilization during the computation: {cpu_usage}%")
This section performs the same matrix multiplication on the CPU, measures the time taken, and estimates CPU utilization.

Results
Time taken for matrix multiplication on GPU: 0.2391 seconds
GPU Utilization: 9.60%
Time taken for matrix multiplication on CPU: 2.0462 seconds
CPU Utilization during computation: 2.1%
Conclusion
The GPU is significantly faster than the CPU for large matrix multiplications, and the GPU is underutilized in this specific task.

References
PyTorch Documentation
NVIDIA GeForce RTX 3060 Specifications



