#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Pure memory bandwidth test kernels
__global__ void copyKernel(float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void vectorAddKernel(float* __restrict__ a, float* __restrict__ b, 
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void computeHeavyKernel(float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Heavy compute - like your kernel
        #pragma unroll 10
        for(int i = 0; i < 100; i++) {
            val = sinf(val) * 2.0f + cosf(val) * 1.5f;
        }
        out[idx] = val;
    }
}

__global__ void scaledCopyKernel(float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = in[idx];
        out[idx] = val;
    }
}

void testBandwidth(const char* name, size_t bytes, float ms) {
    double gb = bytes / (1024.0 * 1024.0 * 1024.0);
    double gb_per_s = (gb * 1000.0) / ms;
    double percent = (gb_per_s / 504.0) * 100.0;  // RTX 4070 max = 504 GB/s
    
    printf("%-20s: %7.2f GB/s (%5.1f%% of peak)\n", name, gb_per_s, percent);
}

int main() {
    const size_t N = 256 * 1024 * 1024;  // 256M elements (1GB)
    const size_t bytes = N * sizeof(float);
    const int ITERATIONS = 100;
    
    printf("=== RTX 4070 Bandwidth Test ===\n");
    printf("Data size: %.2f GB\n", bytes / (1024.0 * 1024.0 * 1024.0));
    printf("Iterations: %d\n\n", ITERATIONS);
    
    // Allocate memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Initialize
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 2, bytes);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for(int i = 0; i < 10; i++) {
        copyKernel<<<grid, block>>>(d_a, d_b, N);
    }
    cudaDeviceSynchronize();
    
    printf("Kernel Bandwidth Tests:\n");
    printf("----------------------------------------\n");
    
    // Test 1: Simple Copy (Read + Write = 2x bandwidth)
    cudaEventRecord(start);
    for(int i = 0; i < ITERATIONS; i++) {
        copyKernel<<<grid, block>>>(d_a, d_b, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    testBandwidth("Simple Copy", 2 * bytes * ITERATIONS, ms1);
    
    // Test 2: Vector Add (Read 2 + Write 1 = 3x bandwidth)
    cudaEventRecord(start);
    for(int i = 0; i < ITERATIONS; i++) {
        vectorAddKernel<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);
    testBandwidth("Vector Add", 3 * bytes * ITERATIONS, ms2);
    
    // Test 3: Compute Heavy (Like your kernel)
    cudaEventRecord(start);
    for(int i = 0; i < ITERATIONS; i++) {
        computeHeavyKernel<<<grid, block>>>(d_a, d_b, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms3;
    cudaEventElapsedTime(&ms3, start, stop);
    testBandwidth("Compute Heavy", 2 * bytes * ITERATIONS, ms3);
    
    // Test 4: Vectorized Copy (float4)
    dim3 grid4((N/4 + block.x - 1) / block.x);
    cudaEventRecord(start);
    for(int i = 0; i < ITERATIONS; i++) {
        scaledCopyKernel<<<grid4, block>>>((float4*)d_a, (float4*)d_b, N/4);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms4;
    cudaEventElapsedTime(&ms4, start, stop);
    testBandwidth("Float4 Copy", 2 * bytes * ITERATIONS, ms4);
    
    printf("\n");
    printf("Analysis:\n");
    printf("----------------------------------------\n");
    printf("Memory-bound (Copy):    %.1f%% bandwidth used\n", (2*bytes*ITERATIONS/ms1/1.024e6*100/504));
    printf("Compute-bound (Heavy):  %.1f%% bandwidth used\n", (2*bytes*ITERATIONS/ms3/1.024e6*100/504));
    printf("Compute/Memory ratio:   %.1fx slower due to compute\n", ms3/ms1);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

// Derleme:
// nvcc -O3 -arch=sm_89 bandwidth_test.cu -o bandwidth_test