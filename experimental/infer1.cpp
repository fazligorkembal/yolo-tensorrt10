#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

// #include "auto/logger.h"
#include "auto/engine.h"

int main()
{
    spdlog::set_level(spdlog::level::info);
    spdlog::debug("Starting the TensorRTForge experimental inference...");

    std::string model_path_onnx = "/home/user/Documents/tensorrtforge/build/yolo11n.onnx";
    std::string image_path = "/home/user/Documents/tensorrtforge/tensorrtforge/input_samples/test_input2.jpg";

    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing
    // calibration data.
    options.calibrationDataDirectoryPath = "";
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;
    // Specify the directory where you want the model engine model file saved.
    options.engineFileDir = ".";
    Engine<float> engine(options);

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;
    if (image.empty())
    {
        spdlog::error("Failed to load image from path: {}", image_path);
        return -1;
    }

    d_image.upload(image);

    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = true;

    bool succ = engine.buildLoadNetwork(model_path_onnx, subVals, divVals, normalize);
    if (!succ)
    {
        throw std::runtime_error("Unable to build or load TensorRT engine.");
    }else{
        spdlog::info("Successfully built and loaded the TensorRT engine.");
    }

    spdlog::debug("Ending the TensorRTForge experimental inference...");
    return 0;
}


//trtexec --loadEngine=yolo11n.engine.NVIDIAGeForceRTX4070.fp16.1.1.-1.-1.-1 --fp16 --iterations=50000 --useSpinWait --infStreams=2 --useCudaGraph
// --useManagedMemory
// --exposeDMA
// --allocationStrategy=spec

/*
    trtexec --loadEngine=yolo11n.engine.NVIDIAGeForceRTX4070.fp16.1.1.-1.-1.-1 --fp16 --iterations=20000 --> 1234.51 qps
    trtexec --loadEngine=yolo11n.engine.NVIDIAGeForceRTX4070.fp16.1.1.-1.-1.-1 --fp16 --iterations=20000 --exposeDMA --> 662.933 qps

*/


// latest trtexec --loadEngine=yolo11n.engine.NVIDIAGeForceRTX4070.fp16.1.1.-1.-1.-1         --fp16         --useCudaGraph  --iterations=20000         --warmUp=400   --infStreams=2 --threads --useSpinWait