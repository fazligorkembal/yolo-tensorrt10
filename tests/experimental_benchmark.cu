#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <functional>
#include <cmath>
#include <mutex>
#include <thread>

// Görüntü boyutları
const int BATCH = 1;
const int CHANNELS = 3;
const int HEIGHT = 640;
const int WIDTH = 640;
const size_t IMAGE_SIZE = BATCH * CHANNELS * HEIGHT * WIDTH * sizeof(float);
const int TOTAL_IMAGES = 6000; // Tüm testlerde aynı toplam resim sayısı

// Ağırlaştırılmış dummy kernel
__global__ void dummyKernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float val = input[idx];
        // Daha fazla işlem ekleyelim
        for (int i = 0; i < 100; i++)
        {
            val = sinf(val) * cosf(val);
        }
        output[idx] = val;
    }
}

// Zaman ölçme fonksiyonu (iterasyonlu)
float measureTime(std::function<void()> func, int iterations)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        func();
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

void runSynchronous()
{
    float *h_input = new float[IMAGE_SIZE / sizeof(float)];
    float *h_output = new float[IMAGE_SIZE / sizeof(float)];
    float *d_input, *d_output;

    cudaMalloc(&d_input, IMAGE_SIZE);
    cudaMalloc(&d_output, IMAGE_SIZE);

    // Veriyi hazırla
    for (size_t i = 0; i < IMAGE_SIZE / sizeof(float); i++)
    {
        h_input[i] = static_cast<float>(i);
    }

    int iterations = TOTAL_IMAGES; // 60,000 iterasyon
    auto time = measureTime([&]()
                            {
        cudaMemcpy(d_input, h_input, IMAGE_SIZE, cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
        dummyKernel<<<gridSize, blockSize>>>(d_input, d_output, IMAGE_SIZE / sizeof(float));
        
        cudaMemcpy(h_output, d_output, IMAGE_SIZE, cudaMemcpyDeviceToHost); }, iterations);

    float throughput = (iterations * 1000.0f) / time; // resim/saniye
    std::cout << "Senkronize - Toplam Süre: " << time << " ms, Throughput: " << throughput << " img/s\n";

    // Temizlik
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

void runAsynchronous()
{
    float *h_input, *h_output;
    cudaHostAlloc(&h_input, IMAGE_SIZE, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, IMAGE_SIZE, cudaHostAllocDefault);

    float *d_input, *d_output;
    cudaMalloc(&d_input, IMAGE_SIZE);
    cudaMalloc(&d_output, IMAGE_SIZE);

    // Veriyi hazırla
    for (size_t i = 0; i < IMAGE_SIZE / sizeof(float); i++)
    {
        h_input[i] = static_cast<float>(i);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int iterations = TOTAL_IMAGES; // 60,000 iterasyon
    auto time = measureTime([&]()
                            {
        cudaMemcpyAsync(d_input, h_input, IMAGE_SIZE, cudaMemcpyHostToDevice, stream);
        
        int blockSize = 256;
        int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
        dummyKernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, IMAGE_SIZE / sizeof(float));
        
        cudaMemcpyAsync(h_output, d_output, IMAGE_SIZE, cudaMemcpyDeviceToHost, stream); }, iterations);

    float throughput = (iterations * 1000.0f) / time;
    std::cout << "Asenkron - Toplam Süre: " << time << " ms, Throughput: " << throughput << " img/s\n";

    // Temizlik
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

void runCUDAGraph()
{
    float *h_input, *h_output;
    cudaHostAlloc(&h_input, IMAGE_SIZE, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, IMAGE_SIZE, cudaHostAllocDefault);

    float *d_input, *d_output;
    cudaMalloc(&d_input, IMAGE_SIZE);
    cudaMalloc(&d_output, IMAGE_SIZE);

    // Veriyi hazırla
    for (size_t i = 0; i < IMAGE_SIZE / sizeof(float); i++)
    {
        h_input[i] = static_cast<float>(i);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // CUDA Graph oluşturma
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(d_input, h_input, IMAGE_SIZE, cudaMemcpyHostToDevice, stream);

    int blockSize = 256;
    int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
    dummyKernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, IMAGE_SIZE / sizeof(float));

    cudaMemcpyAsync(h_output, d_output, IMAGE_SIZE, cudaMemcpyDeviceToHost, stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    int iterations = TOTAL_IMAGES; // 60,000 iterasyon
    auto time = measureTime([&]()
                            { cudaGraphLaunch(graphExec, stream); }, iterations);

    float throughput = (iterations * 1000.0f) / time;
    std::cout << "CUDA Graph - Toplam Süre: " << time << " ms, Throughput: " << throughput << " img/s\n";

    // Temizlik
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

void runMultiStream()
{
    const int IMAGES_PER_ITER = 3;
    const int ITERATIONS = TOTAL_IMAGES / IMAGES_PER_ITER; // 20,000 iterasyon

    float *h_input[IMAGES_PER_ITER], *h_output[IMAGES_PER_ITER];
    float *d_input[IMAGES_PER_ITER], *d_output[IMAGES_PER_ITER];
    cudaStream_t streams[IMAGES_PER_ITER];

    // Bellek ayırma
    for (int i = 0; i < IMAGES_PER_ITER; i++)
    {
        cudaHostAlloc(&h_input[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaHostAlloc(&h_output[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaMalloc(&d_input[i], IMAGE_SIZE);
        cudaMalloc(&d_output[i], IMAGE_SIZE);
        cudaStreamCreate(&streams[i]);

        // Veriyi hazırla
        for (size_t j = 0; j < IMAGE_SIZE / sizeof(float); j++)
        {
            h_input[i][j] = static_cast<float>(j);
        }
    }

    auto time = measureTime([&]()
                            {
        for (int i = 0; i < IMAGES_PER_ITER; i++) {
            cudaMemcpyAsync(d_input[i], h_input[i], IMAGE_SIZE, cudaMemcpyHostToDevice, streams[i]);
            
            int blockSize = 256;
            int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
            dummyKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_input[i], d_output[i], IMAGE_SIZE / sizeof(float));
            
            cudaMemcpyAsync(h_output[i], d_output[i], IMAGE_SIZE, cudaMemcpyDeviceToHost, streams[i]);
        } }, ITERATIONS);

    float throughput = (TOTAL_IMAGES * 1000.0f) / time;
    std::cout << "Çoklu Stream - Toplam Süre: " << time << " ms, Throughput: " << throughput << " img/s\n";

    // Temizlik
    for (int i = 0; i < IMAGES_PER_ITER; i++)
    {
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaStreamDestroy(streams[i]);
    }
}

void runOptimized()
{
    const int IMAGES_PER_ITER = 3;
    const int ITERATIONS = TOTAL_IMAGES / IMAGES_PER_ITER; // 20,000 iterasyon

    float *h_input[IMAGES_PER_ITER], *h_output[IMAGES_PER_ITER];
    float *d_input[IMAGES_PER_ITER], *d_output[IMAGES_PER_ITER];
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamCreate(&stream);

    // Bellek ayırma
    for (int i = 0; i < IMAGES_PER_ITER; i++)
    {
        cudaHostAlloc(&h_input[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaHostAlloc(&h_output[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaMalloc(&d_input[i], IMAGE_SIZE);
        cudaMalloc(&d_output[i], IMAGE_SIZE);

        // Veriyi hazırla
        for (size_t j = 0; j < IMAGE_SIZE / sizeof(float); j++)
        {
            h_input[i][j] = static_cast<float>(j);
        }
    }

    // Tek graph oluştur
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < IMAGES_PER_ITER; i++)
    {
        cudaMemcpyAsync(d_input[i], h_input[i], IMAGE_SIZE, cudaMemcpyHostToDevice, stream);

        int blockSize = 256;
        int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
        dummyKernel<<<gridSize, blockSize, 0, stream>>>(d_input[i], d_output[i], IMAGE_SIZE / sizeof(float));

        cudaMemcpyAsync(h_output[i], d_output[i], IMAGE_SIZE, cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    auto time = measureTime([&]()
                            { cudaGraphLaunch(graphExec, stream); }, ITERATIONS);

    float throughput = (TOTAL_IMAGES * 1000.0f) / time;
    std::cout << "Optimize Edilmiş - Toplam Süre: " << time << " ms, Throughput: " << throughput << " img/s\n";

    // Temizlik
    for (int i = 0; i < IMAGES_PER_ITER; i++)
    {
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
    }
    cudaStreamDestroy(stream);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

void runOptimizedPipeline() {
    const int NUM_STREAMS = 3;
    const int ITERATIONS = 6000 / NUM_STREAMS;
    
    float *h_input[NUM_STREAMS], *h_output[NUM_STREAMS];
    float *d_input[NUM_STREAMS], *d_output[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t startEvents[NUM_STREAMS], endEvents[NUM_STREAMS];
    
    // Bellek ayırma
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaHostAlloc(&h_input[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaHostAlloc(&h_output[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaMalloc(&d_input[i], IMAGE_SIZE);
        cudaMalloc(&d_output[i], IMAGE_SIZE);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&startEvents[i]);
        cudaEventCreate(&endEvents[i]);
        
        // Veriyi hazırla
        for (size_t j = 0; j < IMAGE_SIZE / sizeof(float); j++) {
            h_input[i][j] = static_cast<float>(j);
        }
    }

    // Zaman ölçümü
    cudaEvent_t totalStart, totalEnd;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalEnd);
    cudaEventRecord(totalStart);
    
    // Pipeline başlat
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(d_input[i], h_input[i], IMAGE_SIZE, cudaMemcpyHostToDevice, streams[i]);
        cudaEventRecord(startEvents[i], streams[i]);
    }
    
    // Ana döngü
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            // Kernel çalıştır
            int blockSize = 256;
            int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
            dummyKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_input[i], d_output[i], IMAGE_SIZE / sizeof(float));
            
            // D2H transferini başlat
            cudaMemcpyAsync(h_output[i], d_output[i], IMAGE_SIZE, cudaMemcpyDeviceToHost, streams[i]);
            cudaEventRecord(endEvents[i], streams[i]);
            
            // Bir sonraki iterasyon için H2D transferini başlat
            if (iter < ITERATIONS - 1) {
                // D2H'nin bitmesini beklemeden H2D'yi başlat
                cudaMemcpyAsync(d_input[i], h_input[i], IMAGE_SIZE, cudaMemcpyHostToDevice, streams[i]);
                cudaEventRecord(startEvents[i], streams[i]);
            }
        }
    }
    
    // Tüm stream'lerin bitmesini bekle
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(totalEnd);
    cudaEventSynchronize(totalEnd);
    
    float totalTime;
    cudaEventElapsedTime(&totalTime, totalStart, totalEnd);
    
    float throughput = (6000 * 1000.0f) / totalTime;
    std::cout << "Optimize Pipeline - Toplam Süre: " << totalTime << " ms, Throughput: " << throughput << " img/s\n";
    
    // Temizlik
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(startEvents[i]);
        cudaEventDestroy(endEvents[i]);
    }
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalEnd);
}

void runOptimizedGraph() {
    const int NUM_STREAMS = 3;
    const int ITERATIONS = 6000 / NUM_STREAMS;
    
    float *h_input[NUM_STREAMS], *h_output[NUM_STREAMS];
    float *d_input[NUM_STREAMS], *d_output[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    cudaGraph_t graphs[NUM_STREAMS];
    cudaGraphExec_t graphExecs[NUM_STREAMS];
    
    // Bellek ayırma
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaHostAlloc(&h_input[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaHostAlloc(&h_output[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaMalloc(&d_input[i], IMAGE_SIZE);
        cudaMalloc(&d_output[i], IMAGE_SIZE);
        cudaStreamCreate(&streams[i]);
        
        // Veriyi hazırla
        for (size_t j = 0; j < IMAGE_SIZE / sizeof(float); j++) {
            h_input[i][j] = static_cast<float>(j);
        }
        
        // Her stream için graph oluştur (sadece bir kez!)
        cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeGlobal);
        cudaMemcpyAsync(d_input[i], h_input[i], IMAGE_SIZE, cudaMemcpyHostToDevice, streams[i]);
        
        int blockSize = 256;
        int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
        dummyKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_input[i], d_output[i], IMAGE_SIZE / sizeof(float));
        
        cudaMemcpyAsync(h_output[i], d_output[i], IMAGE_SIZE, cudaMemcpyDeviceToHost, streams[i]);
        cudaStreamEndCapture(streams[i], &graphs[i]);
        cudaGraphInstantiate(&graphExecs[i], graphs[i], NULL, NULL, 0);
    }

    // Zaman ölçümü
    cudaEvent_t totalStart, totalEnd;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalEnd);
    cudaEventRecord(totalStart);
    
    // Ana döngü
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Tüm stream'leri aynı anda başlat
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaGraphLaunch(graphExecs[i], streams[i]);
        }
        
        // Bir sonraki iterasyon için veriyi hazırla
        if (iter < ITERATIONS - 1) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                // Veriyi güncelle (CPU üzerinde)
                for (size_t j = 0; j < IMAGE_SIZE / sizeof(float); j++) {
                    h_input[i][j] = static_cast<float>(j + iter + 1);
                }
            }
        }
    }
    
    // Tüm stream'lerin bitmesini bekle
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(totalEnd);
    cudaEventSynchronize(totalEnd);
    
    float totalTime;
    cudaEventElapsedTime(&totalTime, totalStart, totalEnd);
    
    float throughput = (6000 * 1000.0f) / totalTime;
    std::cout << "Optimize Graph - Toplam Süre: " << totalTime << " ms, Throughput: " << throughput << " img/s\n";
    
    // Temizlik
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaStreamDestroy(streams[i]);
        cudaGraphExecDestroy(graphExecs[i]);
        cudaGraphDestroy(graphs[i]);
    }
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalEnd);
}

// Pinned bellek ile daha hızlı transfer
void optimizeMemoryTransfers() {
    // Unified Memory kullanımı alternatifi
    float *unified_input, *unified_output;
    cudaMallocManaged(&unified_input, IMAGE_SIZE);
    cudaMallocManaged(&unified_output, IMAGE_SIZE);
    
    // Veriyi hazırla
    for (size_t i = 0; i < IMAGE_SIZE / sizeof(float); i++) {
        unified_input[i] = static_cast<float>(i);
    }
    
    // Prefetch ile GPU'ya veriyi önceden getir
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMemPrefetchAsync(unified_input, IMAGE_SIZE, deviceId);
    cudaMemPrefetchAsync(unified_output, IMAGE_SIZE, deviceId);
    
    // Kernel çalıştır
    int blockSize = 256;
    int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
    dummyKernel<<<gridSize, blockSize>>>(unified_input, unified_output, IMAGE_SIZE / sizeof(float));
    
    // Sonuçları CPU'ya geri getir
    cudaMemPrefetchAsync(unified_output, IMAGE_SIZE, cudaCpuDeviceId);
    cudaDeviceSynchronize();
    
    // Temizlik
    cudaFree(unified_input);
    cudaFree(unified_output);
}

void runOptimizedPipelineFixed() {
    const int NUM_STREAMS = 3;
    const int ITERATIONS = 6000 / NUM_STREAMS;
    
    float *h_input[NUM_STREAMS], *h_output[NUM_STREAMS];
    float *d_input[NUM_STREAMS], *d_output[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    
    // Bellek ayırma (event olmadan)
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaHostAlloc(&h_input[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaHostAlloc(&h_output[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaMalloc(&d_input[i], IMAGE_SIZE);
        cudaMalloc(&d_output[i], IMAGE_SIZE);
        cudaStreamCreate(&streams[i]);
        
        // Veriyi hazırla
        for (size_t j = 0; j < IMAGE_SIZE / sizeof(float); j++) {
            h_input[i][j] = static_cast<float>(j);
        }
    }

    // Zaman ölçümü için sadece 2 event
    cudaEvent_t totalStart, totalEnd;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalEnd);
    cudaEventRecord(totalStart);
    
    // İlk iterasyon için H2D transferlerini başlat
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(d_input[i], h_input[i], IMAGE_SIZE, cudaMemcpyHostToDevice, streams[i]);
    }
    
    // Ana döngü - event kullanmadan
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            // Kernel çalıştır
            int blockSize = 256;
            int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
            dummyKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_input[i], d_output[i], IMAGE_SIZE / sizeof(float));
            
            // D2H transferini başlat
            cudaMemcpyAsync(h_output[i], d_output[i], IMAGE_SIZE, cudaMemcpyDeviceToHost, streams[i]);
            
            // Bir sonraki iterasyon için H2D transferini başlat
            if (iter < ITERATIONS - 1) {
                // Event beklemeden doğrudan başlat
                cudaMemcpyAsync(d_input[i], h_input[i], IMAGE_SIZE, cudaMemcpyHostToDevice, streams[i]);
            }
        }
    }
    
    // Tüm stream'lerin bitmesini bekle
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(totalEnd);
    cudaEventSynchronize(totalEnd);
    
    float totalTime;
    cudaEventElapsedTime(&totalTime, totalStart, totalEnd);
    
    float throughput = (6000 * 1000.0f) / totalTime;
    std::cout << "Optimize Pipeline (Fixed) - Toplam Süre: " << totalTime << " ms, Throughput: " << throughput << " img/s\n";
    
    // Temizlik
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalEnd);
}





std::mutex mtx;

void processStream(int streamId, float* h_input, float* h_output, float* d_input, float* d_output, 
                  cudaStream_t stream, int iterations, int* counter) {
    for (int iter = 0; iter < iterations; iter++) {
        // H2D transferi
        cudaMemcpyAsync(d_input, h_input, IMAGE_SIZE, cudaMemcpyHostToDevice, stream);
        
        // Kernel çalıştır
        int blockSize = 256;
        int gridSize = (IMAGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
        dummyKernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, IMAGE_SIZE / sizeof(float));
        
        // D2H transferi
        cudaMemcpyAsync(h_output, d_output, IMAGE_SIZE, cudaMemcpyDeviceToHost, stream);
        
        // İlerlemeyi güncelle
        {
            std::lock_guard<std::mutex> lock(mtx);
            (*counter)++;
        }
    }
}

void runMultiThreadedPipeline() {
    const int NUM_STREAMS = 6; // Stream sayısını artırdık
    const int ITERATIONS = 6000 / NUM_STREAMS;
    
    float *h_input[NUM_STREAMS], *h_output[NUM_STREAMS];
    float *d_input[NUM_STREAMS], *d_output[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    
    // Bellek ayırma
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaHostAlloc(&h_input[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaHostAlloc(&h_output[i], IMAGE_SIZE, cudaHostAllocDefault);
        cudaMalloc(&d_input[i], IMAGE_SIZE);
        cudaMalloc(&d_output[i], IMAGE_SIZE);
        cudaStreamCreate(&streams[i]);
        
        // Veriyi hazırla
        for (size_t j = 0; j < IMAGE_SIZE / sizeof(float); j++) {
            h_input[i][j] = static_cast<float>(j);
        }
    }

    // Zaman ölçümü
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Thread'leri oluştur
    std::vector<std::thread> threads;
    int counter = 0;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        threads.emplace_back(processStream, i, h_input[i], h_output[i], 
                            d_input[i], d_output[i], streams[i], ITERATIONS, &counter);
    }
    
    // Thread'lerin bitmesini bekle
    for (auto& t : threads) {
        t.join();
    }
    
    // Tüm stream'lerin bitmesini bekle
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);
    
    float throughput = (6000 * 1000.0f) / totalTime;
    std::cout << "Multi-Threaded Pipeline - Toplam Süre: " << totalTime << " ms, Throughput: " << throughput << " img/s\n";
    std::cout << "Toplam işlenen resim: " << counter << "\n";
    
    // Temizlik
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void runBatchProcessing() {
    const int BATCH_SIZE = 4; // Bir seferde 4 resim işle
    const int NUM_BATCHES = 6000 / BATCH_SIZE;
    
    float *h_input, *h_output;
    float *d_input, *d_output;
    cudaStream_t stream;
    
    // Bellek ayırma (batch boyutunda)
    size_t batch_size = BATCH_SIZE * IMAGE_SIZE;
    cudaHostAlloc(&h_input, batch_size, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, batch_size, cudaHostAllocDefault);
    cudaMalloc(&d_input, batch_size);
    cudaMalloc(&d_output, batch_size);
    cudaStreamCreate(&stream);
    
    // Veriyi hazırla
    for (size_t i = 0; i < batch_size / sizeof(float); i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Zaman ölçümü
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Ana döngü - batch işleme
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        // H2D transferi (batch)
        cudaMemcpyAsync(d_input, h_input, batch_size, cudaMemcpyHostToDevice, stream);
        
        // Kernel çalıştır (batch)
        int blockSize = 256;
        int gridSize = (batch_size / sizeof(float) + blockSize - 1) / blockSize;
        dummyKernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, batch_size / sizeof(float));
        
        // D2H transferi (batch)
        cudaMemcpyAsync(h_output, d_output, batch_size, cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);
    
    float throughput = (6000 * 1000.0f) / totalTime;
    std::cout << "Batch Processing - Toplam Süre: " << totalTime << " ms, Throughput: " << throughput << " img/s\n";
    
    // Temizlik
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    //runSynchronous();
    //runAsynchronous();
    //runCUDAGraph();
    //runMultiStream();
    //runOptimized();
    //optimizeMemoryTransfers();
    //runOptimizedPipeline();
    //runOptimizedGraph();
    //runOptimizedPipelineFixed();


    std::cout << "=== Multi-Threaded Pipeline Testi ===\n";
    runMultiThreadedPipeline();
    
    std::cout << "\n=== Batch Processing Testi ===\n";
    runBatchProcessing();
    

    return 0;
}