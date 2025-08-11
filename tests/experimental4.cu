#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <chrono>

// Input dimensions: 1 x 3 x 640 x 640 (NCHW format)
constexpr int BATCH = 1;
constexpr int CHANNELS = 3;
constexpr int HEIGHT = 640;
constexpr int WIDTH = 640;
constexpr int TOTAL_SIZE = BATCH * CHANNELS * HEIGHT * WIDTH;
constexpr int ITERATIONS = 60000;

// Optimized image processing kernel
__global__ void imageKernel(float* __restrict__ d_input, 
                            float* __restrict__ d_output, 
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = d_input[idx];
        
        // Simulated image processing operations
        #pragma unroll 4
        for(int i = 0; i < 100; i++) {
            val = sinf(val) * 2.0f + cosf(val) * 1.5f;
            val = val * 0.99f + 0.01f;
        }
        
        d_output[idx] = val;
    }
}

// 2D convolution-like kernel for better cache utilization
__global__ void convKernel(float* __restrict__ d_input,
                          float* __restrict__ d_output,
                          int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x < width && y < height && c < channels) {
        int idx = c * width * height + y * width + x;
        float val = d_input[idx];
        
        // Heavy compute to simulate real workload
        #pragma unroll 8
        for(int i = 0; i < 50; i++) {
            val = __fmaf_rn(val, 1.01f, sinf(val));
        }
        
        d_output[idx] = val;
    }
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "=== Optimized CUDA Kernel for 1x3x640x640 ===" << std::endl;
    std::cout << "Input shape: " << BATCH << "x" << CHANNELS << "x" << HEIGHT << "x" << WIDTH << std::endl;
    std::cout << "Total elements: " << TOTAL_SIZE << " (" << TOTAL_SIZE * sizeof(float) / (1024.0f * 1024.0f) << " MB)" << std::endl;
    std::cout << "Iterations: " << ITERATIONS << std::endl << std::endl;
    
    size_t bytes = TOTAL_SIZE * sizeof(float);
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, imageKernel, 0, 0);
    std::cout << "Optimal block size: " << blockSize << std::endl;

    // Pinned memory allocation
    float *h_input, *h_output;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    CHECK_CUDA(cudaMallocHost(&h_output, bytes));
    
    // Device memory
    float *d_input, *d_output, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, bytes));
    
    // Initialize input (simulate image data)
    for(int i = 0; i < TOTAL_SIZE; i++) {
        h_input[i] = static_cast<float>(i % 256) / 255.0f;
    }
    
    // Create stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Async transfer to device (only once!)
    CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream));
    
    // Kernel configurations
    // For 1D kernel
    const int BLOCK_SIZE = 768;
    const int GRID_SIZE = (TOTAL_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // For 2D kernel (better for image data)
    dim3 blockSize2D(32, 32, 1);
    dim3 gridSize2D((WIDTH + blockSize2D.x - 1) / blockSize2D.x,
                    (HEIGHT + blockSize2D.y - 1) / blockSize2D.y,
                    CHANNELS);
    
    // Create CUDA Graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // Start graph capture
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Add kernels to graph - alternating between two kernels for variety
    imageKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_input, d_temp, TOTAL_SIZE);
    convKernel<<<gridSize2D, blockSize2D, 0, stream>>>(d_temp, d_output, WIDTH, HEIGHT, CHANNELS);
    
    // End graph capture
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    
    // Instantiate graph
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    
    // Warmup (important for accurate timing)
    for(int i = 0; i < 100; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Performance measurement
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    
    CHECK_CUDA(cudaEventRecord(start_event, stream));
    
    // Main iteration loop - 6000 iterations
    for(int iter = 0; iter < ITERATIONS; iter++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    
    CHECK_CUDA(cudaEventRecord(stop_event, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Calculate elapsed time
    float gpu_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event));
    
    // Transfer result back (only once at the end!)
    CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Performance results
    std::cout << "=== PERFORMANS SONUÇLARI ===" << std::endl;
    std::cout << "GPU süresi: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "İterasyon sayısı: " << ITERATIONS << std::endl;
    std::cout << "İterasyon başına süre: " << gpu_time_ms / ITERATIONS << " ms" << std::endl;
    std::cout << "Throughput: " << (ITERATIONS * 1000.0f) / gpu_time_ms << " iter/s" << std::endl;
    
    // Calculate bandwidth (2 kernels, each reads and writes)
    double gb_processed = (4.0 * TOTAL_SIZE * sizeof(float) * ITERATIONS) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "İşlenen veri: " << gb_processed << " GB" << std::endl;
    std::cout << "Efektif bant genişliği: " << (gb_processed * 1000.0) / gpu_time_ms << " GB/s" << std::endl;
    
    // Image processing metrics
    double megapixels = (WIDTH * HEIGHT * ITERATIONS) / 1e6;
    std::cout << "İşlenen frame sayısı: " << ITERATIONS << std::endl;
    std::cout << "FPS: " << (ITERATIONS * 1000.0f) / gpu_time_ms << " frame/s" << std::endl;
    std::cout << "Megapixel/saniye: " << (megapixels * 1000.0) / gpu_time_ms << " MP/s" << std::endl;
    
    // Verify results
    bool valid = true;
    int errors = 0;
    for(int i = 0; i < 1000; i++) {
        if(std::isnan(h_output[i]) || std::isinf(h_output[i])) {
            valid = false;
            errors++;
        }
    }
    std::cout << "\nSonuç doğrulaması: " << (valid ? "BAŞARILI" : "BAŞARISIZ") << std::endl;
    if(!valid) std::cout << "Hata sayısı: " << errors << std::endl;
    
    // Get GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp));
    
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    
    std::cout << "\nProgram başarıyla tamamlandı!" << std::endl;
    
    return 0;
}

// Derleme:
// nvcc -O3 -arch=sm_89 -use_fast_math optimized_kernel.cu -o optimized_kernel
// 
// Maksimum performans için:
// nvcc -O3 -arch=sm_89 -use_fast_math -Xptxas -O3 -Xcompiler -O3 optimized_kernel.cu -o optimized_kernel