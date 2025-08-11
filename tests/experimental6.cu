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
constexpr int NUM_STREAMS = 2;

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
    std::cout << "=== Dual Stream CUDA Kernel for 1x3x640x640 ===" << std::endl;
    std::cout << "Input shape: " << BATCH << "x" << CHANNELS << "x" << HEIGHT << "x" << WIDTH << std::endl;
    std::cout << "Total elements: " << TOTAL_SIZE << " (" << TOTAL_SIZE * sizeof(float) / (1024.0f * 1024.0f) << " MB)" << std::endl;
    std::cout << "Iterations: " << ITERATIONS << " (split across " << NUM_STREAMS << " streams)" << std::endl;
    std::cout << "Iterations per stream: " << ITERATIONS/NUM_STREAMS << std::endl << std::endl;
    
    size_t bytes = TOTAL_SIZE * sizeof(float);
    
    // Get optimal block size
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, imageKernel, 0, 0);
    std::cout << "Optimal block size: " << blockSize << std::endl;
    
    // Set L2 cache config for better performance
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    // Pinned memory allocation
    float *h_input, *h_output;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    CHECK_CUDA(cudaMallocHost(&h_output, bytes));
    
    // Device memory - separate buffers for each stream
    float *d_inputs[NUM_STREAMS];
    float *d_outputs[NUM_STREAMS]; 
    float *d_temps[NUM_STREAMS];
    
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMalloc(&d_inputs[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_outputs[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_temps[i], bytes));
    }
    
    // Initialize input (simulate image data)
    for(int i = 0; i < TOTAL_SIZE; i++) {
        h_input[i] = static_cast<float>(i % 256) / 255.0f;
    }
    
    // Create streams with high priority
    cudaStream_t streams[NUM_STREAMS];
    int priority_high, priority_low;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, priority_high));
    }
    
    // Transfer data to all stream buffers
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_inputs[i], h_input, bytes, 
                                   cudaMemcpyHostToDevice, streams[i]));
    }
    
    // Kernel configurations
    const int BLOCK_SIZE = blockSize;
    const int GRID_SIZE = (TOTAL_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 blockSize2D(32, 32, 1);
    dim3 gridSize2D((WIDTH + blockSize2D.x - 1) / blockSize2D.x,
                    (HEIGHT + blockSize2D.y - 1) / blockSize2D.y,
                    CHANNELS);
    
    // Create CUDA Graphs for each stream
    cudaGraph_t graphs[NUM_STREAMS];
    cudaGraphExec_t graphExecs[NUM_STREAMS];
    
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaStreamBeginCapture(streams[s], cudaStreamCaptureModeGlobal));
        
        // Add kernels to graph
        imageKernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams[s]>>>(
            d_inputs[s], d_temps[s], TOTAL_SIZE);
        convKernel<<<gridSize2D, blockSize2D, 0, streams[s]>>>(
            d_temps[s], d_outputs[s], WIDTH, HEIGHT, CHANNELS);
        
        CHECK_CUDA(cudaStreamEndCapture(streams[s], &graphs[s]));
        CHECK_CUDA(cudaGraphInstantiate(&graphExecs[s], graphs[s], NULL, NULL, 0));
    }
    
    // Warmup - run each graph multiple times
    for(int iter = 0; iter < 100; iter++) {
        for(int s = 0; s < NUM_STREAMS; s++) {
            CHECK_CUDA(cudaGraphLaunch(graphExecs[s], streams[s]));
        }
    }
    
    // Synchronize all streams after warmup
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
    }
    
    // Performance measurement
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    
    CHECK_CUDA(cudaEventRecord(start_event, streams[0]));
    
    // Main iteration loop - interleaved execution
    const int iters_per_stream = ITERATIONS / NUM_STREAMS;
    
    for(int iter = 0; iter < iters_per_stream; iter++) {
        // Launch graphs on all streams for maximum overlap
        for(int s = 0; s < NUM_STREAMS; s++) {
            CHECK_CUDA(cudaGraphLaunch(graphExecs[s], streams[s]));
        }
        
        // Optional: Add periodic sync to prevent queue buildup
        if(iter % 500 == 499) {
            for(int s = 0; s < NUM_STREAMS; s++) {
                CHECK_CUDA(cudaStreamSynchronize(streams[s]));
            }
        }
    }
    
    // Record end event on last stream
    CHECK_CUDA(cudaEventRecord(stop_event, streams[NUM_STREAMS-1]));
    
    // Synchronize all streams
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
    }
    
    // Calculate elapsed time
    float gpu_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event));
    
    // Transfer results back from first stream buffer
    CHECK_CUDA(cudaMemcpyAsync(h_output, d_outputs[0], bytes, 
                               cudaMemcpyDeviceToHost, streams[0]));
    CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    
    // Performance results
    std::cout << "\n=== PERFORMANS SONUÇLARI ===" << std::endl;
    std::cout << "GPU süresi: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Toplam iterasyon: " << ITERATIONS << std::endl;
    std::cout << "İterasyon başına süre: " << gpu_time_ms / ITERATIONS << " ms" << std::endl;
    std::cout << "Throughput: " << (ITERATIONS * 1000.0f) / gpu_time_ms << " iter/s" << std::endl;
    
    // Calculate effective operations (both streams working)
    double total_ops = 2.0 * iters_per_stream;  // 2 streams
    std::cout << "Stream başına iterasyon: " << iters_per_stream << std::endl;
    std::cout << "Paralel stream kullanımı: " << NUM_STREAMS << std::endl;
    
    // Calculate bandwidth
    double gb_processed = (4.0 * TOTAL_SIZE * sizeof(float) * ITERATIONS) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "İşlenen veri: " << gb_processed << " GB" << std::endl;
    std::cout << "Efektif bant genişliği: " << (gb_processed * 1000.0) / gpu_time_ms << " GB/s" << std::endl;
    
    // Image processing metrics
    double megapixels = (WIDTH * HEIGHT * ITERATIONS) / 1e6;
    std::cout << "İşlenen frame sayısı: " << ITERATIONS << std::endl;
    std::cout << "FPS: " << (ITERATIONS * 1000.0f) / gpu_time_ms << " frame/s" << std::endl;
    std::cout << "Megapixel/saniye: " << (megapixels * 1000.0) / gpu_time_ms << " MP/s" << std::endl;
    
    // Stream efficiency calculation
    float single_stream_estimate = gpu_time_ms * NUM_STREAMS;
    float efficiency = (single_stream_estimate / gpu_time_ms - 1.0f) * 100.0f;
    std::cout << "\nStream paralelizm verimliliği: " << efficiency << "%" << std::endl;
    
    // Verify results
    bool valid = true;
    int errors = 0;
    for(int i = 0; i < 1000; i++) {
        if(std::isnan(h_output[i]) || std::isinf(h_output[i])) {
            valid = false;
            errors++;
        }
    }
    std::cout << "Sonuç doğrulaması: " << (valid ? "BAŞARILI" : "BAŞARISIZ") << std::endl;
    if(!valid) std::cout << "Hata sayısı: " << errors << std::endl;
    
    // Get GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaGraphDestroy(graphs[s]));
        CHECK_CUDA(cudaGraphExecDestroy(graphExecs[s]));
        CHECK_CUDA(cudaStreamDestroy(streams[s]));
        CHECK_CUDA(cudaFree(d_inputs[s]));
        CHECK_CUDA(cudaFree(d_outputs[s]));
        CHECK_CUDA(cudaFree(d_temps[s]));
    }
    
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    
    std::cout << "\nProgram başarıyla tamamlandı!" << std::endl;
    
    return 0;
}

// Derleme:
// nvcc -O3 -arch=sm_89 -use_fast_math dual_stream_kernel.cu -o dual_stream_kernel
// 
// Maksimum performans için:
// nvcc -O3 -arch=sm_89 -use_fast_math -Xptxas -O3,-dlcm=ca -Xcompiler -O3 dual_stream_kernel.cu -o dual_stream_kernel