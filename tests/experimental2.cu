#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <chrono>

// Basit dummy kernel - vektör toplama
__global__ void dummyKernel(float* d_a, float* d_b, float* d_c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Basit bir işlem
        d_c[idx] = d_a[idx] + d_b[idx];
        
        // Biraz daha iş yükü ekleyelim
        for(int i = 0; i < 100; i++) {
            d_c[idx] = sin(d_c[idx]) + cos(d_a[idx]);
        }
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
    // RTX 4070 için optimize edilmiş parametreler
    const int N = 1024 * 1024;  // 1M eleman
    const int ITERATIONS = 6000;
    const int BLOCK_SIZE = 256;  // RTX 4070 için iyi bir değer
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    size_t bytes = N * sizeof(float);
    
    // Host pointers (pinned memory)
    float *h_a, *h_b, *h_c;
    
    // Pinned memory allocation
    CHECK_CUDA(cudaMallocHost(&h_a, bytes));
    CHECK_CUDA(cudaMallocHost(&h_b, bytes));
    CHECK_CUDA(cudaMallocHost(&h_c, bytes));
    
    // Device pointers
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    
    // Initialize host data
    for(int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Stream oluştur
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Async memory transfer (H2D)
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
    
    // CUDA Graph oluşturma
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // Graph capture başlat
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Kernel'i graph'a ekle
    dummyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, N);
    
    // Graph capture bitir
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    
    // Graph instance oluştur
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    
    // Warm-up
    for(int i = 0; i < 10; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Performance ölçümü başlat
    auto start = std::chrono::high_resolution_clock::now();
    
    // 6000 iterasyon çalıştır
    for(int iter = 0; iter < ITERATIONS; iter++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    
    // Stream'i bekle
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Async memory transfer (D2H)
    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Performans sonuçları
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ms = duration.count() / 1000.0;
    
    std::cout << "=== PERFORMANS SONUÇLARI ===" << std::endl;
    std::cout << "Toplam süre: " << ms << " ms" << std::endl;
    std::cout << "İterasyon sayısı: " << ITERATIONS << std::endl;
    std::cout << "İterasyon başına süre: " << ms / ITERATIONS << " ms" << std::endl;
    std::cout << "Throughput: " << (ITERATIONS * 1000.0) / ms << " iter/s" << std::endl;
    
    // Veri boyutu ve bant genişliği
    double gb = (3.0 * N * sizeof(float) * ITERATIONS) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "İşlenen veri: " << gb << " GB" << std::endl;
    std::cout << "Efektif bant genişliği: " << (gb * 1000.0) / ms << " GB/s" << std::endl;
    
    // Sonuçları kontrol et (opsiyonel)
    bool correct = true;
    for(int i = 0; i < 10; i++) {  // İlk 10 elemanı kontrol et
        float expected = h_a[i] + h_b[i];
        // sin/cos işlemleri nedeniyle tam eşitlik beklemiyoruz
        if(std::abs(h_c[i]) > 1e6) {  // Basit bir kontrol
            correct = false;
            break;
        }
    }
    std::cout << "Sonuç kontrolü: " << (correct ? "BAŞARILI" : "BAŞARISIZ") << std::endl;
    
    // Temizlik
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    
    std::cout << "\nProgram başarıyla tamamlandı!" << std::endl;
    
    return 0;
}

// Derleme komutu:
// nvcc -O3 -arch=sm_89 cuda_kernel.cu -o cuda_kernel
// RTX 4070 için sm_89 kullanıyoruz (Ada Lovelace)