#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

constexpr int TOTAL_SIZE = 1228800;  // 640x640x3
constexpr int ITERATIONS = 1000;     // Daha az iterasyon
constexpr int NUM_STREAMS = 2;
constexpr int COMPUTE_INTENSITY = 5000;  // ÇOK daha ağır compute

// GERÇEKTEN ağır kernel
__global__ void heavyKernel(float* __restrict__ input, 
                            float* __restrict__ output, 
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        float acc = 0.0f;
        
        // SUPER heavy compute - GPU'yu gerçekten meşgul et
        #pragma unroll 2
        for(int i = 0; i < COMPUTE_INTENSITY; i++) {
            // Complex math operations
            val = __fmaf_rn(val, 1.00001f, 0.00001f);
            val = __sinf(val) + __cosf(val * 2.0f);
            val = __expf(-val * val) + __logf(fabsf(val) + 1.0f);
            acc += val;
            
            // Add some divergent branches
            if(i % 100 == 0) {
                val = sqrtf(fabsf(val));
            }
        }
        
        output[idx] = val + acc * 0.00001f;
    }
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                     << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "=== Heavy Kernel Multi-Stream Test ===" << std::endl;
    std::cout << "Data size: " << TOTAL_SIZE << " elements" << std::endl;
    std::cout << "Compute intensity: " << COMPUTE_INTENSITY << " ops/element" << std::endl;
    std::cout << "Iterations: " << ITERATIONS << std::endl;
    std::cout << "Streams: " << NUM_STREAMS << std::endl << std::endl;
    
    size_t bytes = TOTAL_SIZE * sizeof(float);
    
    // Allocate memory
    float *h_input, *h_output;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    CHECK_CUDA(cudaMallocHost(&h_output, bytes));
    
    float *d_inputs[NUM_STREAMS];
    float *d_outputs[NUM_STREAMS];
    
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMalloc(&d_inputs[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_outputs[i], bytes));
    }
    
    // Initialize
    for(int i = 0; i < TOTAL_SIZE; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }
    
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // Copy data
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_inputs[i], h_input, bytes,
                                   cudaMemcpyHostToDevice, streams[i]));
    }
    
    // Kernel config
    int blockSize = 256;
    int gridSize = (TOTAL_SIZE + blockSize - 1) / blockSize;
    
    // Create graphs
    cudaGraph_t graphs[NUM_STREAMS];
    cudaGraphExec_t graphExecs[NUM_STREAMS];
    
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaStreamBeginCapture(streams[s], cudaStreamCaptureModeGlobal));
        
        // Single heavy kernel per graph
        heavyKernel<<<gridSize, blockSize, 0, streams[s]>>>(
            d_inputs[s], d_outputs[s], TOTAL_SIZE);
        
        CHECK_CUDA(cudaStreamEndCapture(streams[s], &graphs[s]));
        CHECK_CUDA(cudaGraphInstantiate(&graphExecs[s], graphs[s], NULL, NULL, 0));
    }
    
    // Warmup
    for(int i = 0; i < 5; i++) {
        for(int s = 0; s < NUM_STREAMS; s++) {
            CHECK_CUDA(cudaGraphLaunch(graphExecs[s], streams[s]));
        }
    }
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
    }
    
    // Test 1: Single Stream Performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < ITERATIONS; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExecs[0], streams[0]));
    }
    CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    
    auto end = std::chrono::high_resolution_clock::now();
    double single_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Test 2: Multi-Stream Performance
    start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < ITERATIONS/NUM_STREAMS; i++) {
        // Launch all streams
        for(int s = 0; s < NUM_STREAMS; s++) {
            CHECK_CUDA(cudaGraphLaunch(graphExecs[s], streams[s]));
        }
    }
    
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
    }
    
    end = std::chrono::high_resolution_clock::now();
    double multi_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Results
    std::cout << "=== RESULTS ===" << std::endl;
    std::cout << "Single Stream: " << single_ms << " ms" << std::endl;
    std::cout << "Multi Stream:  " << multi_ms << " ms" << std::endl;
    std::cout << "Speedup: " << single_ms / multi_ms << "x" << std::endl;
    std::cout << "Efficiency: " << (single_ms / multi_ms) / NUM_STREAMS * 100 << "%" << std::endl;
    
    if(multi_ms < single_ms * 0.9) {
        std::cout << "✅ Multi-stream WORKS! Parallelism achieved." << std::endl;
    } else {
        std::cout << "❌ No parallelism. Kernel too light or GPU saturated." << std::endl;
    }
    
    // Cleanup
    for(int s = 0; s < NUM_STREAMS; s++) {
        CHECK_CUDA(cudaGraphDestroy(graphs[s]));
        CHECK_CUDA(cudaGraphExecDestroy(graphExecs[s]));
        CHECK_CUDA(cudaStreamDestroy(streams[s]));
        CHECK_CUDA(cudaFree(d_inputs[s]));
        CHECK_CUDA(cudaFree(d_outputs[s]));
    }
    
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    
    return 0;
}

// nvcc -O3 -arch=sm_89 heavy_kernel_test.cu -o heavy_test