#pragma once

#include "scratch/logging.h"
#include "scratch/cuda_utils.h"
#include "scratch/postprocess.h"
#include "scratch/preprocess.h"
#include "scratch/utils.h"

#include "model_base.hpp"

using namespace nvinfer1;

class ModelDetectionScratch : public ModelBase
{
public:
    explicit ModelDetectionScratch(const Options options) : ModelBase(options)
    {
        // todo: add int8 support
        ASSERT(options_.optimization_type == OptimizationType::fp32 ||
                   options_.optimization_type == OptimizationType::fp16,
               "Optimization type must be fp32 or fp16 for scratch conversion, int8 is not supported yet");

        std::string wts_path = options_.model_path;
        std::string engine_path = wts_path.substr(0, wts_path.find_last_of('.')) + ".engine";

        if (options_.model_path.ends_with(".wts"))
        {
            if (std::filesystem::exists(engine_path))
            {
                std::cout << "Engine file already exists, skipping conversion" << std::endl;
            }
            else
            {
                std::string type;
                float gd = 0.0f;
                float gw = 0.0f;
                int max_channels = 0;

                parse_options(type, gd, gw, max_channels);

                serialize(wts_path, engine_path, type, gd, gw, max_channels);
            }
        }
        else if (options_.model_path.ends_with(".engine") || options_.model_path.ends_with(".plan"))
        {
            engine_path = options_.model_path;
        }
        else
        {
            ASSERT(false, "Model path must be a .wts, .engine or .plan file for scratch conversion");
        }

        deserialize(engine_path);
        prepare_buffer(options_.task_type);
    }
    ~ModelDetectionScratch() override = default;
    void infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch) override;

    void infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch, std::vector<cv::Mat> &masks) override
    {
        ASSERT(false, "This method is not supported for detection task, please use infer without masks");
    }

private:
    void serialize(std::string &wts_name, std::string &engine_path, std::string &type, float &gd, float &gw, int &max_channels) override;
    void deserialize(std::string &engine_path) override;
    void prepare_buffer(TaskType task_type) override;
    void parse_options(std::string &type, float &gd, float &gw, int &max_channels) override;

    Logger gLogger;
    const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    cudaStream_t stream;

    int batch_size = 0;
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;
};