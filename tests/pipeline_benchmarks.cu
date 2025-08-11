// ultimate_tensorrt_pipeline.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cstring>
#include <iomanip>
#include <cuda_runtime.h>
#include <NvInfer.h>

using namespace nvinfer1;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Renkli output için
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

// ============================================================================
// CUDA Kernels - Must be defined outside of classes
// ============================================================================
__global__ void dummyKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        // Simulate some computation
        float val = input[idx];
        val = val * 2.0f + 1.0f;
        val = sqrtf(val);
        val = val * val;
        output[idx % (n/4)] = val;
    }
}

// ============================================================================
// STEP 1: Baseline - Sadece Pinned Memory
// ============================================================================
class Step1_PinnedMemory {
protected:
    size_t inputSize;
    size_t outputSize;
    float* h_input_pinned;
    float* h_output_pinned;
    float* d_input;
    float* d_output;
    cudaStream_t stream;
    
public:
    Step1_PinnedMemory(size_t inSize, size_t outSize) 
        : inputSize(inSize), outputSize(outSize) {
        
        // Pinned memory allocation
        CHECK_CUDA(cudaHostAlloc(&h_input_pinned, inputSize, cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_output_pinned, outputSize, cudaHostAllocDefault));
        
        // Device memory
        CHECK_CUDA(cudaMalloc(&d_input, inputSize));
        CHECK_CUDA(cudaMalloc(&d_output, outputSize));
        
        // Stream
        CHECK_CUDA(cudaStreamCreate(&stream));
        
        std::cout << GREEN << "[Step 1] Pinned Memory initialized" << RESET << std::endl;
    }
    
    virtual ~Step1_PinnedMemory() {
        cudaFreeHost(h_input_pinned);
        cudaFreeHost(h_output_pinned);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }
    
    virtual void processFrame(float* input, float* output) {
        // Copy to pinned memory
        memcpy(h_input_pinned, input, inputSize);
        
        // H2D transfer
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input_pinned, inputSize, 
                                   cudaMemcpyHostToDevice, stream));
        
        // Dummy kernel (TensorRT inference yerine)
        launchDummyKernel(d_input, d_output, stream);
        
        // D2H transfer
        CHECK_CUDA(cudaMemcpyAsync(h_output_pinned, d_output, outputSize, 
                                   cudaMemcpyDeviceToHost, stream));
        
        // Wait
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Copy from pinned memory
        memcpy(output, h_output_pinned, outputSize);
    }
    
protected:
    void launchDummyKernel(float* input, float* output, cudaStream_t stream);
    
    // Kernel will be defined outside the class
};

// Implementation of launchDummyKernel
void Step1_PinnedMemory::launchDummyKernel(float* input, float* output, cudaStream_t stream) {
    // Simulate TensorRT inference
    int threads = 256;
    int blocks = 1024;
    dummyKernel<<<blocks, threads, 0, stream>>>(input, output, inputSize/sizeof(float));
}

// ============================================================================
// STEP 2: Double Buffering
// ============================================================================
class Step2_DoubleBuffering : public Step1_PinnedMemory {
protected:
    // Second set of buffers
    float* h_input_pinned2;
    float* h_output_pinned2;
    float* d_input2;
    float* d_output2;
    
    int currentBuffer = 0;
    
public:
    Step2_DoubleBuffering(size_t inSize, size_t outSize) 
        : Step1_PinnedMemory(inSize, outSize) {
        
        // Allocate second buffer set
        CHECK_CUDA(cudaHostAlloc(&h_input_pinned2, inputSize, cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_output_pinned2, outputSize, cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&d_input2, inputSize));
        CHECK_CUDA(cudaMalloc(&d_output2, outputSize));
        
        std::cout << GREEN << "[Step 2] Double Buffering initialized" << RESET << std::endl;
    }
    
    ~Step2_DoubleBuffering() {
        cudaFreeHost(h_input_pinned2);
        cudaFreeHost(h_output_pinned2);
        cudaFree(d_input2);
        cudaFree(d_output2);
    }
    
    void processFrame(float* input, float* output) override {
        // Select buffers
        float* h_in = (currentBuffer == 0) ? h_input_pinned : h_input_pinned2;
        float* h_out = (currentBuffer == 0) ? h_output_pinned : h_output_pinned2;
        float* d_in = (currentBuffer == 0) ? d_input : d_input2;
        float* d_out = (currentBuffer == 0) ? d_output : d_output2;
        
        // Process with selected buffer
        memcpy(h_in, input, inputSize);
        
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, inputSize, 
                                   cudaMemcpyHostToDevice, stream));
        launchDummyKernel(d_in, d_out, stream);
        CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, outputSize, 
                                   cudaMemcpyDeviceToHost, stream));
        
        CHECK_CUDA(cudaStreamSynchronize(stream));
        memcpy(output, h_out, outputSize);
        
        // Switch buffer
        currentBuffer = 1 - currentBuffer;
    }
};

// ============================================================================
// STEP 3: Event-based Async Pipeline
// ============================================================================
class Step3_AsyncPipeline : public Step2_DoubleBuffering {
protected:
    cudaEvent_t h2d_done[2];
    cudaEvent_t kernel_done[2];
    cudaEvent_t d2h_done[2];
    
    bool firstFrame = true;
    
public:
    Step3_AsyncPipeline(size_t inSize, size_t outSize) 
        : Step2_DoubleBuffering(inSize, outSize) {
        
        // Create events for both buffers
        for(int i = 0; i < 2; i++) {
            CHECK_CUDA(cudaEventCreate(&h2d_done[i]));
            CHECK_CUDA(cudaEventCreate(&kernel_done[i]));
            CHECK_CUDA(cudaEventCreate(&d2h_done[i]));
        }
        
        std::cout << GREEN << "[Step 3] Async Pipeline with Events initialized" << RESET << std::endl;
    }
    
    ~Step3_AsyncPipeline() {
        for(int i = 0; i < 2; i++) {
            cudaEventDestroy(h2d_done[i]);
            cudaEventDestroy(kernel_done[i]);
            cudaEventDestroy(d2h_done[i]);
        }
    }
    
    void processFrameAsync(float* input, int frameId) {
        int buf = frameId % 2;
        
        float* h_in = (buf == 0) ? h_input_pinned : h_input_pinned2;
        float* h_out = (buf == 0) ? h_output_pinned : h_output_pinned2;
        float* d_in = (buf == 0) ? d_input : d_input2;
        float* d_out = (buf == 0) ? d_output : d_output2;
        
        // Wait for previous frame in this buffer to complete
        if(frameId >= 2) {
            CHECK_CUDA(cudaEventSynchronize(d2h_done[buf]));
        }
        
        // Copy input to pinned memory
        memcpy(h_in, input, inputSize);
        
        // Launch pipeline stages with events
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, inputSize, 
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaEventRecord(h2d_done[buf], stream));
        
        launchDummyKernel(d_in, d_out, stream);
        CHECK_CUDA(cudaEventRecord(kernel_done[buf], stream));
        
        CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, outputSize, 
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaEventRecord(d2h_done[buf], stream));
    }
    
    void waitForFrame(float* output, int frameId) {
        int buf = frameId % 2;
        float* h_out = (buf == 0) ? h_output_pinned : h_output_pinned2;
        
        CHECK_CUDA(cudaEventSynchronize(d2h_done[buf]));
        memcpy(output, h_out, outputSize);
    }
};

// ============================================================================
// STEP 4: Multi-Stream
// ============================================================================
class Step4_MultiStream : public Step3_AsyncPipeline {
protected:
    static const int NUM_STREAMS = 3;
    cudaStream_t streams[NUM_STREAMS];
    cudaStream_t h2d_stream;
    cudaStream_t kernel_stream;
    cudaStream_t d2h_stream;
    
public:
    Step4_MultiStream(size_t inSize, size_t outSize) 
        : Step3_AsyncPipeline(inSize, outSize) {
        
        // Create dedicated streams for each stage
        CHECK_CUDA(cudaStreamCreate(&h2d_stream));
        CHECK_CUDA(cudaStreamCreate(&kernel_stream));
        CHECK_CUDA(cudaStreamCreate(&d2h_stream));
        
        for(int i = 0; i < NUM_STREAMS; i++) {
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
        }
        
        std::cout << GREEN << "[Step 4] Multi-Stream initialized" << RESET << std::endl;
    }
    
    ~Step4_MultiStream() {
        cudaStreamDestroy(h2d_stream);
        cudaStreamDestroy(kernel_stream);
        cudaStreamDestroy(d2h_stream);
        
        for(int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
    
    void processFrameMultiStream(float* input, int frameId) {
        int buf = frameId % 2;
        
        float* h_in = (buf == 0) ? h_input_pinned : h_input_pinned2;
        float* h_out = (buf == 0) ? h_output_pinned : h_output_pinned2;
        float* d_in = (buf == 0) ? d_input : d_input2;
        float* d_out = (buf == 0) ? d_output : d_output2;
        
        // Wait for previous frame in this buffer
        if(frameId >= 2) {
            CHECK_CUDA(cudaEventSynchronize(d2h_done[buf]));
        }
        
        memcpy(h_in, input, inputSize);
        
        // H2D on dedicated stream
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, inputSize, 
                                   cudaMemcpyHostToDevice, h2d_stream));
        CHECK_CUDA(cudaEventRecord(h2d_done[buf], h2d_stream));
        
        // Kernel waits for H2D and runs on different stream
        CHECK_CUDA(cudaStreamWaitEvent(kernel_stream, h2d_done[buf]));
        launchDummyKernel(d_in, d_out, kernel_stream);
        CHECK_CUDA(cudaEventRecord(kernel_done[buf], kernel_stream));
        
        // D2H waits for kernel and runs on another stream
        CHECK_CUDA(cudaStreamWaitEvent(d2h_stream, kernel_done[buf]));
        CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, outputSize, 
                                   cudaMemcpyDeviceToHost, d2h_stream));
        CHECK_CUDA(cudaEventRecord(d2h_done[buf], d2h_stream));
    }
};

// ============================================================================
// STEP 5: Threading + Worker Pool
// ============================================================================
class Step5_Threading : public Step4_MultiStream {
protected:
    std::thread workerThread;
    std::queue<std::pair<float*, int>> inputQueue;
    std::queue<std::pair<float*, int>> outputQueue;
    std::mutex inputMutex, outputMutex;
    std::condition_variable inputCV, outputCV;
    std::atomic<bool> stopWorker{false};
    
public:
    Step5_Threading(size_t inSize, size_t outSize) 
        : Step4_MultiStream(inSize, outSize) {
        
        // Start worker thread
        workerThread = std::thread(&Step5_Threading::workerLoop, this);
        
        std::cout << GREEN << "[Step 5] Threading initialized" << RESET << std::endl;
    }
    
    ~Step5_Threading() {
        stopWorker = true;
        inputCV.notify_all();
        if(workerThread.joinable()) {
            workerThread.join();
        }
    }
    
    void submitFrameAsync(float* input, int frameId) {
        // Allocate buffer for input copy
        float* inputCopy = new float[inputSize / sizeof(float)];
        memcpy(inputCopy, input, inputSize);
        
        {
            std::lock_guard<std::mutex> lock(inputMutex);
            inputQueue.push({inputCopy, frameId});
        }
        inputCV.notify_one();
    }
    
    bool getCompletedFrame(float* output, int& frameId) {
        std::unique_lock<std::mutex> lock(outputMutex);
        if(outputQueue.empty()) {
            return false;
        }
        
        auto [outputData, id] = outputQueue.front();
        outputQueue.pop();
        
        memcpy(output, outputData, outputSize);
        delete[] outputData;
        frameId = id;
        
        return true;
    }
    
private:
    void workerLoop() {
        while(!stopWorker) {
            std::unique_lock<std::mutex> lock(inputMutex);
            inputCV.wait(lock, [this] { return !inputQueue.empty() || stopWorker; });
            
            if(stopWorker) break;
            
            auto [input, frameId] = inputQueue.front();
            inputQueue.pop();
            lock.unlock();
            
            // Process frame
            processFrameMultiStream(input, frameId);
            
            // Wait for completion and add to output queue
            int buf = frameId % 2;
            CHECK_CUDA(cudaEventSynchronize(d2h_done[buf]));
            
            float* h_out = (buf == 0) ? h_output_pinned : h_output_pinned2;
            float* outputCopy = new float[outputSize / sizeof(float)];
            memcpy(outputCopy, h_out, outputSize);
            
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                outputQueue.push({outputCopy, frameId});
            }
            outputCV.notify_one();
            
            delete[] input;
        }
    }
};

// ============================================================================
// STEP 6: CUDA Graph Integration
// ============================================================================
class Step6_CudaGraph : public Step5_Threading {
protected:
    cudaGraph_t graph[2];
    cudaGraphExec_t graphExec[2];
    bool graphCreated[2] = {false, false};
    
public:
    Step6_CudaGraph(size_t inSize, size_t outSize) 
        : Step5_Threading(inSize, outSize) {
        
        // Create graphs for both buffers
        for(int buf = 0; buf < 2; buf++) {
            createGraph(buf);
        }
        
        std::cout << GREEN << "[Step 6] CUDA Graph initialized" << RESET << std::endl;
    }
    
    ~Step6_CudaGraph() {
        for(int i = 0; i < 2; i++) {
            if(graphCreated[i]) {
                cudaGraphExecDestroy(graphExec[i]);
                cudaGraphDestroy(graph[i]);
            }
        }
    }
    
    void createGraph(int bufferIndex) {
        float* h_in = (bufferIndex == 0) ? h_input_pinned : h_input_pinned2;
        float* h_out = (bufferIndex == 0) ? h_output_pinned : h_output_pinned2;
        float* d_in = (bufferIndex == 0) ? d_input : d_input2;
        float* d_out = (bufferIndex == 0) ? d_output : d_output2;
        
        // Create a dedicated stream for graph capture
        cudaStream_t captureStream;
        CHECK_CUDA(cudaStreamCreate(&captureStream));
        
        // Start capture
        CHECK_CUDA(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));
        
        // Record all operations
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, inputSize, 
                                   cudaMemcpyHostToDevice, captureStream));
        launchDummyKernel(d_in, d_out, captureStream);
        CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, outputSize, 
                                   cudaMemcpyDeviceToHost, captureStream));
        
        // End capture
        CHECK_CUDA(cudaStreamEndCapture(captureStream, &graph[bufferIndex]));
        
        // Create executable graph
        CHECK_CUDA(cudaGraphInstantiate(&graphExec[bufferIndex], graph[bufferIndex], 
                                        nullptr, nullptr, 0));
        
        graphCreated[bufferIndex] = true;
        cudaStreamDestroy(captureStream);
    }
    
    void processFrameWithGraph(float* input, int frameId) {
        int buf = frameId % 2;
        
        float* h_in = (buf == 0) ? h_input_pinned : h_input_pinned2;
        float* h_out = (buf == 0) ? h_output_pinned : h_output_pinned2;
        
        // Copy input to pinned memory
        memcpy(h_in, input, inputSize);
        
        // Launch graph
        CHECK_CUDA(cudaGraphLaunch(graphExec[buf], streams[frameId % NUM_STREAMS]));
        CHECK_CUDA(cudaEventRecord(d2h_done[buf], streams[frameId % NUM_STREAMS]));
    }
};

// ============================================================================
// ULTIMATE: Tüm Optimizasyonlar Birlikte
// ============================================================================
class UltimateTensorRTPipeline : public Step6_CudaGraph {
private:
    // Triple buffering for ultimate performance
    float* h_input_pinned3;
    float* h_output_pinned3;
    float* d_input3;
    float* d_output3;
    
    // More events for fine control
    cudaEvent_t events[3][3]; // [buffer][stage]
    
    // Performance metrics
    std::atomic<int> framesProcessed{0};
    std::atomic<int> framesInFlight{0};
    
    // Multiple worker threads
    std::vector<std::thread> workers;
    static const int NUM_WORKERS = 2;
    
public:
    UltimateTensorRTPipeline(size_t inSize, size_t outSize) 
        : Step6_CudaGraph(inSize, outSize) {
        
        // Third buffer for triple buffering
        CHECK_CUDA(cudaHostAlloc(&h_input_pinned3, inputSize, cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_output_pinned3, outputSize, cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&d_input3, inputSize));
        CHECK_CUDA(cudaMalloc(&d_output3, outputSize));
        
        // Create events for all buffers and stages
        for(int buf = 0; buf < 3; buf++) {
            for(int stage = 0; stage < 3; stage++) {
                CHECK_CUDA(cudaEventCreate(&events[buf][stage]));
            }
        }
        
        // Start multiple worker threads
        for(int i = 0; i < NUM_WORKERS; i++) {
            workers.emplace_back(&UltimateTensorRTPipeline::ultimateWorkerLoop, this, i);
        }
        
        std::cout << GREEN << "[ULTIMATE] All optimizations enabled!" << RESET << std::endl;
        std::cout << "  ✓ Pinned Memory" << std::endl;
        std::cout << "  ✓ Triple Buffering" << std::endl;
        std::cout << "  ✓ Event-based Async Pipeline" << std::endl;
        std::cout << "  ✓ Multi-Stream (" << NUM_STREAMS << " streams)" << std::endl;
        std::cout << "  ✓ Multi-Threading (" << NUM_WORKERS << " workers)" << std::endl;
        std::cout << "  ✓ CUDA Graphs" << std::endl;
    }
    
    ~UltimateTensorRTPipeline() {
        stopWorker = true;
        inputCV.notify_all();
        
        for(auto& worker : workers) {
            if(worker.joinable()) {
                worker.join();
            }
        }
        
        cudaFreeHost(h_input_pinned3);
        cudaFreeHost(h_output_pinned3);
        cudaFree(d_input3);
        cudaFree(d_output3);
        
        for(int buf = 0; buf < 3; buf++) {
            for(int stage = 0; stage < 3; stage++) {
                cudaEventDestroy(events[buf][stage]);
            }
        }
    }
    
    void processUltimate(float* input, int frameId) {
        int buf = frameId % 3;  // Triple buffering
        int streamId = frameId % NUM_STREAMS;
        
        // Select buffer set
        float* h_in, *h_out, *d_in, *d_out;
        switch(buf) {
            case 0:
                h_in = h_input_pinned; h_out = h_output_pinned;
                d_in = d_input; d_out = d_output;
                break;
            case 1:
                h_in = h_input_pinned2; h_out = h_output_pinned2;
                d_in = d_input2; d_out = d_output2;
                break;
            case 2:
                h_in = h_input_pinned3; h_out = h_output_pinned3;
                d_in = d_input3; d_out = d_output3;
                break;
        }
        
        // Wait for previous frame in this buffer (if any)
        if(frameId >= 3) {
            CHECK_CUDA(cudaEventSynchronize(events[buf][2]));
        }
        
        // Async copy to pinned memory (can overlap with GPU work)
        std::thread([this, h_in, input]() {
            memcpy(h_in, input, inputSize);
        }).detach();
        
        // Use graph if available for this buffer size
        if(buf < 2 && graphCreated[buf]) {
            // Graph path (fastest)
            CHECK_CUDA(cudaGraphLaunch(graphExec[buf], streams[streamId]));
            CHECK_CUDA(cudaEventRecord(events[buf][2], streams[streamId]));
        } else {
            // Manual pipeline with events
            CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, inputSize, 
                                       cudaMemcpyHostToDevice, h2d_stream));
            CHECK_CUDA(cudaEventRecord(events[buf][0], h2d_stream));
            
            CHECK_CUDA(cudaStreamWaitEvent(kernel_stream, events[buf][0]));
            launchDummyKernel(d_in, d_out, kernel_stream);
            CHECK_CUDA(cudaEventRecord(events[buf][1], kernel_stream));
            
            CHECK_CUDA(cudaStreamWaitEvent(d2h_stream, events[buf][1]));
            CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, outputSize, 
                                       cudaMemcpyDeviceToHost, d2h_stream));
            CHECK_CUDA(cudaEventRecord(events[buf][2], d2h_stream));
        }
        
        framesInFlight++;
    }
    
private:
    void ultimateWorkerLoop(int workerId) {
        while(!stopWorker) {
            std::unique_lock<std::mutex> lock(inputMutex);
            inputCV.wait(lock, [this] { return !inputQueue.empty() || stopWorker; });
            
            if(stopWorker) break;
            
            auto [input, frameId] = inputQueue.front();
            inputQueue.pop();
            lock.unlock();
            
            // Process with all optimizations
            processUltimate(input, frameId);
            
            // Handle completion asynchronously
            int buf = frameId % 3;
            std::thread([this, buf, frameId]() {
                CHECK_CUDA(cudaEventSynchronize(events[buf][2]));
                
                // Get output buffer
                float* h_out;
                switch(buf) {
                    case 0: h_out = h_output_pinned; break;
                    case 1: h_out = h_output_pinned2; break;
                    case 2: h_out = h_output_pinned3; break;
                }
                
                float* outputCopy = new float[outputSize / sizeof(float)];
                memcpy(outputCopy, h_out, outputSize);
                
                {
                    std::lock_guard<std::mutex> lock(outputMutex);
                    outputQueue.push({outputCopy, frameId});
                }
                outputCV.notify_one();
                
                framesProcessed++;
                framesInFlight--;
            }).detach();
            
            delete[] input;
        }
    }
    
public:
    void getPerformanceStats() {
        std::cout << CYAN << "\n=== Performance Stats ===" << RESET << std::endl;
        std::cout << "Frames Processed: " << framesProcessed.load() << std::endl;
        std::cout << "Frames In Flight: " << framesInFlight.load() << std::endl;
    }
};

// ============================================================================
// Benchmark Runner
// ============================================================================
class BenchmarkRunner {
private:
    size_t inputSize = 1 * 640 * 640;  // 10 MB
    size_t outputSize = 1 * 640 * 640;  // 1 MB
    int numFrames = 20000;
    
    template<typename T>
    double benchmarkPipeline(const std::string& name) {
        std::cout << YELLOW << "\nBenchmarking: " << name << RESET << std::endl;
        
        T pipeline(inputSize, outputSize);
        
        // Prepare test data
        std::vector<float*> inputs;
        std::vector<float*> outputs;
        for(int i = 0; i < numFrames; i++) {
            float* input = new float[inputSize / sizeof(float)];
            float* output = new float[outputSize / sizeof(float)];
            
            // Fill with test data
            for(size_t j = 0; j < inputSize / sizeof(float); j++) {
                input[j] = i + j * 0.001f;
            }
            
            inputs.push_back(input);
            outputs.push_back(output);
        }
        
        // Warmup
        for(int i = 0; i < 5; i++) {
            pipeline.processFrame(inputs[i], outputs[i]);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for(int i = 0; i < numFrames; i++) {
            pipeline.processFrame(inputs[i], outputs[i]);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        // Cleanup
        for(auto ptr : inputs) delete[] ptr;
        for(auto ptr : outputs) delete[] ptr;
        
        double throughput = (numFrames * 1000.0) / elapsed.count();
        std::cout << "Time: " << elapsed.count() << " ms" << std::endl;
        std::cout << "Throughput: " << throughput << " frames/sec" << std::endl;
        
        return elapsed.count();
    }
    
public:
    void runAllBenchmarks() {
        std::cout << MAGENTA << "\n========================================" << std::endl;
        std::cout << "     ULTIMATE TENSORRT BENCHMARK" << std::endl;
        std::cout << "========================================" << RESET << std::endl;
        std::cout << "Input Size: " << inputSize / (1024*1024) << " MB" << std::endl;
        std::cout << "Output Size: " << outputSize / (1024*1024) << " MB" << std::endl;
        std::cout << "Number of Frames: " << numFrames << std::endl;
        
        std::vector<std::pair<std::string, double>> results;
        
        // Step by step benchmarks
        results.push_back({"Step 1: Pinned Memory", 
                          benchmarkPipeline<Step1_PinnedMemory>("Step 1: Pinned Memory")});
        
        results.push_back({"Step 2: Double Buffering", 
                          benchmarkPipeline<Step2_DoubleBuffering>("Step 2: Double Buffering")});
        
        // Advanced benchmarks
        std::cout << YELLOW << "\nBenchmarking: Step 3: Async Pipeline" << RESET << std::endl;
        Step3_AsyncPipeline asyncPipeline(inputSize, outputSize);
        
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < numFrames; i++) {
            float* input = new float[inputSize / sizeof(float)];
            for(size_t j = 0; j < inputSize / sizeof(float); j++) {
                input[j] = i + j * 0.001f;
            }
            asyncPipeline.processFrameAsync(input, i);
            delete[] input;
        }
        
        // Wait for all frames
        for(int i = 0; i < numFrames; i++) {
            float* output = new float[outputSize / sizeof(float)];
            asyncPipeline.waitForFrame(output, i);
            delete[] output;
        }
        auto asyncTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "Time: " << asyncTime << " ms" << std::endl;
        std::cout << "Throughput: " << (numFrames * 1000.0 / asyncTime) << " frames/sec" << std::endl;
        results.push_back({"Step 3: Async Pipeline", asyncTime});
        
        // Multi-stream benchmark
        std::cout << YELLOW << "\nBenchmarking: Step 4: Multi-Stream" << RESET << std::endl;
        Step4_MultiStream multiStream(inputSize, outputSize);
        
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < numFrames; i++) {
            float* input = new float[inputSize / sizeof(float)];
            for(size_t j = 0; j < inputSize / sizeof(float); j++) {
                input[j] = i + j * 0.001f;
            }
            multiStream.processFrameMultiStream(input, i);
            delete[] input;
        }
        
        cudaDeviceSynchronize();
        auto multiStreamTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "Time: " << multiStreamTime << " ms" << std::endl;
        std::cout << "Throughput: " << (numFrames * 1000.0 / multiStreamTime) << " frames/sec" << std::endl;
        results.push_back({"Step 4: Multi-Stream", multiStreamTime});
        
        // Threading benchmark
        std::cout << YELLOW << "\nBenchmarking: Step 5: Threading" << RESET << std::endl;
        Step5_Threading threading(inputSize, outputSize);
        
        start = std::chrono::high_resolution_clock::now();
        
        // Submit all frames
        for(int i = 0; i < numFrames; i++) {
            float* input = new float[inputSize / sizeof(float)];
            for(size_t j = 0; j < inputSize / sizeof(float); j++) {
                input[j] = i + j * 0.001f;
            }
            threading.submitFrameAsync(input, i);
            delete[] input;
        }
        
        // Collect results
        int collected = 0;
        while(collected < numFrames) {
            float* output = new float[outputSize / sizeof(float)];
            int frameId;
            if(threading.getCompletedFrame(output, frameId)) {
                collected++;
            }
            delete[] output;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        auto threadingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "Time: " << threadingTime << " ms" << std::endl;
        std::cout << "Throughput: " << (numFrames * 1000.0 / threadingTime) << " frames/sec" << std::endl;
        results.push_back({"Step 5: Threading", threadingTime});
        
        // CUDA Graph benchmark
        std::cout << YELLOW << "\nBenchmarking: Step 6: CUDA Graph" << RESET << std::endl;
        Step6_CudaGraph cudaGraph(inputSize, outputSize);
        
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < numFrames; i++) {
            float* input = new float[inputSize / sizeof(float)];
            for(size_t j = 0; j < inputSize / sizeof(float); j++) {
                input[j] = i + j * 0.001f;
            }
            cudaGraph.processFrameWithGraph(input, i);
            delete[] input;
        }
        
        cudaDeviceSynchronize();
        auto graphTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "Time: " << graphTime << " ms" << std::endl;
        std::cout << "Throughput: " << (numFrames * 1000.0 / graphTime) << " frames/sec" << std::endl;
        results.push_back({"Step 6: CUDA Graph", graphTime});
        
        // ULTIMATE benchmark
        std::cout << YELLOW << "\nBenchmarking: ULTIMATE Pipeline" << RESET << std::endl;
        UltimateTensorRTPipeline ultimate(inputSize, outputSize);
        
        start = std::chrono::high_resolution_clock::now();
        
        // Submit all frames
        for(int i = 0; i < numFrames; i++) {
            float* input = new float[inputSize / sizeof(float)];
            for(size_t j = 0; j < inputSize / sizeof(float); j++) {
                input[j] = i + j * 0.001f;
            }
            ultimate.submitFrameAsync(input, i);
            delete[] input;
        }
        
        // Collect results
        collected = 0;
        while(collected < numFrames) {
            float* output = new float[outputSize / sizeof(float)];
            int frameId;
            if(ultimate.getCompletedFrame(output, frameId)) {
                collected++;
            }
            delete[] output;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        auto ultimateTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "Time: " << ultimateTime << " ms" << std::endl;
        std::cout << "Throughput: " << (numFrames * 1000.0 / ultimateTime) << " frames/sec" << std::endl;
        results.push_back({"ULTIMATE", ultimateTime});
        
        ultimate.getPerformanceStats();
        
        // Print summary
        printSummary(results);
    }
    
private:
    void printSummary(const std::vector<std::pair<std::string, double>>& results) {
        std::cout << GREEN << "\n========================================" << std::endl;
        std::cout << "           BENCHMARK SUMMARY" << std::endl;
        std::cout << "========================================" << RESET << std::endl;
        
        double baseline = results[0].second;
        
        std::cout << std::setw(30) << std::left << "Optimization"
                  << std::setw(12) << "Time (ms)"
                  << std::setw(15) << "Throughput"
                  << std::setw(10) << "Speedup" << std::endl;
        std::cout << std::string(67, '-') << std::endl;
        
        for(const auto& [name, time] : results) {
            double throughput = numFrames * 1000.0 / time;
            double speedup = baseline / time;
            
            // Color code based on speedup
            std::string color = RESET;
            if(speedup > 5.0) color = MAGENTA;
            else if(speedup > 3.0) color = GREEN;
            else if(speedup > 1.5) color = CYAN;
            else if(speedup > 1.0) color = YELLOW;
            
            std::cout << color
                      << std::setw(30) << std::left << name
                      << std::setw(12) << std::fixed << std::setprecision(2) << time
                      << std::setw(15) << std::fixed << std::setprecision(1) 
                      << (std::to_string(throughput) + " fps")
                      << std::setw(10) << std::fixed << std::setprecision(2) 
                      << (std::to_string(speedup) + "x")
                      << RESET << std::endl;
        }
        
        // Best result
        auto best = std::min_element(results.begin(), results.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.second < b.second;
                                     });
        
        std::cout << std::endl;
        std::cout << MAGENTA << "🏆 WINNER: " << best->first << RESET << std::endl;
        std::cout << "   Final Speedup: " << std::fixed << std::setprecision(2) 
                  << (baseline / best->second) << "x faster than baseline!" << std::endl;
        std::cout << "   Throughput: " << std::fixed << std::setprecision(1)
                  << (numFrames * 1000.0 / best->second) << " frames/second" << std::endl;
        
        // Calculate theoretical limits
        std::cout << CYAN << "\n=== Theoretical Limits ===" << RESET << std::endl;
        
        // PCIe bandwidth (Gen4 x16)
        double pcieBandwidth = 31.5; // GB/s bidirectional
        double dataPerFrame = (inputSize + outputSize) / (1024.0 * 1024.0 * 1024.0); // GB
        double maxPCIeThroughput = pcieBandwidth / dataPerFrame;
        
        std::cout << "PCIe 4.0 x16 Limit: " << std::fixed << std::setprecision(1) 
                  << maxPCIeThroughput << " frames/sec" << std::endl;
        std::cout << "Current Utilization: " << std::fixed << std::setprecision(1)
                  << ((numFrames * 1000.0 / best->second) / maxPCIeThroughput * 100.0) 
                  << "%" << std::endl;
    }
};

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char** argv) {
    // GPU Info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << CYAN << "=== System Information ===" << RESET << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Memory Clock: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Peak Memory Bandwidth: " 
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
              << " GB/s" << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
    
    // Run benchmarks
    BenchmarkRunner runner;
    runner.runAllBenchmarks();
    
    // Performance tips
    std::cout << YELLOW << "\n=== Performance Tips ===" << RESET << std::endl;
    std::cout << "1. Use NVIDIA Nsight Systems for detailed profiling:" << std::endl;
    std::cout << "   nsys profile --stats=true ./ultimate_tensorrt_pipeline" << std::endl;
    std::cout << "2. Monitor GPU utilization:" << std::endl;
    std::cout << "   nvidia-smi dmon -s pucvmet" << std::endl;
    std::cout << "3. Check PCIe bandwidth:" << std::endl;
    std::cout << "   nvidia-smi -q -d PCIE" << std::endl;
    std::cout << "4. Enable GPU boost:" << std::endl;
    std::cout << "   sudo nvidia-smi -lgc <max_clock>" << std::endl;
    
    return 0;
}