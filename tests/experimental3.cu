#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// Input dimensions: 1 x 3 x 640 x 640 (NCHW format)
constexpr int BATCH = 1;
constexpr int CHANNELS = 3;
constexpr int HEIGHT = 640;
constexpr int WIDTH = 640;
constexpr int TOTAL_SIZE = BATCH * CHANNELS * HEIGHT * WIDTH;

// Multi-stream parametreleri
constexpr int NUM_STREAMS = 4;  // RTX 4070 için optimal
constexpr int ITERATIONS = 6000;

// Image processing kernel - her channel için ayrı işlem
__global__ void imageProcessingKernel(float* input, float* output, int channel_offset, int pixels_per_channel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < pixels_per_channel) {
        int global_idx = channel_offset + idx;
        
        // Dummy image processing işlemleri
        float val = input[global_idx];
        
        // Normalization (0-255 to 0-1)
        val = val / 255.0f;
        
        // Contrast adjustment
        val = (val - 0.5f) * 1.2f + 0.5f;
        
        // Gamma correction
        val = powf(val, 1.0f/2.2f);
        
        // Edge enhancement simulation
        float edge = sinf(val * 3.14159f) * 0.1f;
        val = fmaxf(0.0f, fminf(1.0f, val + edge));
        
        // Denormalize
        output[global_idx] = val * 255.0f;
    }
}

// Convolution-like kernel (3x3 filter simulation)
__global__ void convolutionKernel(float* input, float* output, int channel, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int base_idx = channel * height * width;
        int center_idx = base_idx + y * width + x;
        
        // Simple 3x3 blur filter
        float sum = 0.0f;
        for(int dy = -1; dy <= 1; dy++) {
            for(int dx = -1; dx <= 1; dx++) {
                int idx = base_idx + (y + dy) * width + (x + dx);
                sum += input[idx];
            }
        }
        
        output[center_idx] = sum / 9.0f;
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

// Stream başına work chunk hesapla
struct StreamWork {
    int channel;
    int offset;
    int size;
};

int main() {
    std::cout << "=== CUDA Multi-Stream Image Processing ===" << std::endl;
    std::cout << "Input shape: " << BATCH << "x" << CHANNELS << "x" << HEIGHT << "x" << WIDTH << std::endl;
    std::cout << "Total size: " << TOTAL_SIZE << " elements (" << TOTAL_SIZE * sizeof(float) / (1024.0f * 1024.0f) << " MB)" << std::endl;
    std::cout << "Streams: " << NUM_STREAMS << std::endl;
    std::cout << "Iterations: " << ITERATIONS << std::endl << std::endl;
    
    size_t total_bytes = TOTAL_SIZE * sizeof(float);
    int pixels_per_channel = HEIGHT * WIDTH;
    
    // Pinned memory allocation
    float *h_input, *h_output, *h_temp;
    CHECK_CUDA(cudaMallocHost(&h_input, total_bytes));
    CHECK_CUDA(cudaMallocHost(&h_output, total_bytes));
    CHECK_CUDA(cudaMallocHost(&h_temp, total_bytes));
    
    // Device memory - her stream için ayrı buffer
    std::vector<float*> d_inputs(NUM_STREAMS);
    std::vector<float*> d_outputs(NUM_STREAMS);
    std::vector<float*> d_temps(NUM_STREAMS);
    
    // Her stream bir channel'ı işleyecek (3 channel, 4 stream için bir stream 2 pass yapacak)
    size_t bytes_per_channel = pixels_per_channel * sizeof(float);
    
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMalloc(&d_inputs[i], bytes_per_channel));
        CHECK_CUDA(cudaMalloc(&d_outputs[i], bytes_per_channel));
        CHECK_CUDA(cudaMalloc(&d_temps[i], bytes_per_channel));
    }
    
    // Initialize input data (simulate image data)
    for(int i = 0; i < TOTAL_SIZE; i++) {
        h_input[i] = static_cast<float>((i % 256));  // 0-255 range
    }
    
    // Create streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // CUDA Graphs - her stream için ayrı graph
    std::vector<cudaGraph_t> graphs(NUM_STREAMS);
    std::vector<cudaGraphExec_t> graphExecs(NUM_STREAMS);
    
    // Graph oluşturma - her stream için
    for(int stream_id = 0; stream_id < NUM_STREAMS; stream_id++) {
        CHECK_CUDA(cudaStreamBeginCapture(streams[stream_id], cudaStreamCaptureModeGlobal));
        
        // Kernel launch parametreleri
        dim3 blockSize1D(256);
        dim3 gridSize1D((pixels_per_channel + blockSize1D.x - 1) / blockSize1D.x);
        
        dim3 blockSize2D(16, 16);
        dim3 gridSize2D((WIDTH + blockSize2D.x - 1) / blockSize2D.x,
                       (HEIGHT + blockSize2D.y - 1) / blockSize2D.y);
        
        // İki aşamalı processing
        imageProcessingKernel<<<gridSize1D, blockSize1D, 0, streams[stream_id]>>>(
            d_inputs[stream_id], d_temps[stream_id], 0, pixels_per_channel
        );
        
        convolutionKernel<<<gridSize2D, blockSize2D, 0, streams[stream_id]>>>(
            d_temps[stream_id], d_outputs[stream_id], 0, HEIGHT, WIDTH
        );
        
        CHECK_CUDA(cudaStreamEndCapture(streams[stream_id], &graphs[stream_id]));
        CHECK_CUDA(cudaGraphInstantiate(&graphExecs[stream_id], graphs[stream_id], NULL, NULL, 0));
    }
    
    // Warm-up
    for(int stream_id = 0; stream_id < NUM_STREAMS; stream_id++) {
        int channel = stream_id % CHANNELS;
        float* src = h_input + channel * pixels_per_channel;
        
        CHECK_CUDA(cudaMemcpyAsync(d_inputs[stream_id], src, bytes_per_channel, 
                                   cudaMemcpyHostToDevice, streams[stream_id]));
        CHECK_CUDA(cudaGraphLaunch(graphExecs[stream_id], streams[stream_id]));
    }
    
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    // Performance measurement
    auto start = std::chrono::high_resolution_clock::now();
    
    // Ana işlem döngüsü - 6000 iterasyon
    for(int iter = 0; iter < ITERATIONS; iter++) {
        // Her iterasyonda tüm channel'ları işle
        for(int channel = 0; channel < CHANNELS; channel++) {
            int stream_id = channel % NUM_STREAMS;
            
            // Async H2D transfer
            float* src = h_input + channel * pixels_per_channel;
            CHECK_CUDA(cudaMemcpyAsync(d_inputs[stream_id], src, bytes_per_channel,
                                       cudaMemcpyHostToDevice, streams[stream_id]));
            
            // Graph execution
            CHECK_CUDA(cudaGraphLaunch(graphExecs[stream_id], streams[stream_id]));
            
            // Async D2H transfer
            float* dst = h_output + channel * pixels_per_channel;
            CHECK_CUDA(cudaMemcpyAsync(dst, d_outputs[stream_id], bytes_per_channel,
                                       cudaMemcpyDeviceToHost, streams[stream_id]));
        }
        
        // Periyodik senkronizasyon (her 100 iterasyonda)
        if(iter % 100 == 99) {
            for(int i = 0; i < NUM_STREAMS; i++) {
                CHECK_CUDA(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    
    // Final synchronization
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Performance results
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ms = duration.count() / 1000.0;
    double seconds = ms / 1000.0;
    
    std::cout << "=== PERFORMANS SONUÇLARI ===" << std::endl;
    std::cout << "Toplam süre: " << ms << " ms (" << seconds << " saniye)" << std::endl;
    std::cout << "İterasyon sayısı: " << ITERATIONS << std::endl;
    std::cout << "İterasyon başına süre: " << ms / ITERATIONS << " ms" << std::endl;
    std::cout << "FPS (throughput): " << (ITERATIONS / seconds) << " frame/s" << std::endl;
    
    // Data throughput
    double gb_processed = (TOTAL_SIZE * sizeof(float) * ITERATIONS * 2) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "\nİşlenen toplam veri: " << gb_processed << " GB" << std::endl;
    std::cout << "Efektif bant genişliği: " << (gb_processed / seconds) << " GB/s" << std::endl;
    
    // Pixel throughput
    double megapixels = (HEIGHT * WIDTH * CHANNELS * ITERATIONS) / 1e6;
    std::cout << "İşlenen megapixel: " << megapixels << " MP" << std::endl;
    std::cout << "Megapixel/saniye: " << (megapixels / seconds) << " MP/s" << std::endl;
    
    // Verify results (basic check)
    bool valid = true;
    for(int i = 0; i < 100; i++) {
        if(std::isnan(h_output[i]) || std::isinf(h_output[i])) {
            valid = false;
            break;
        }
    }
    std::cout << "\nSonuç doğrulaması: " << (valid ? "BAŞARILI" : "BAŞARISIZ") << std::endl;
    
    // Cleanup
    for(int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaGraphDestroy(graphs[i]));
        CHECK_CUDA(cudaGraphExecDestroy(graphExecs[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaFree(d_inputs[i]));
        CHECK_CUDA(cudaFree(d_outputs[i]));
        CHECK_CUDA(cudaFree(d_temps[i]));
    }
    
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFreeHost(h_temp));
    
    // Device reset for profiling
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "\nProgram başarıyla tamamlandı!" << std::endl;
    
    return 0;
}

// Derleme komutları:
// nvcc -O3 -arch=sm_89 multi_stream_kernel.cu -o multi_stream_kernel
// 
// Profiling için:
// nvprof ./multi_stream_kernel
// veya
// ncu --set full ./multi_stream_kernel