#pragma once

#include "scratch/logging.h"
#include "scratch/cuda_utils.h"
#include "scratch/postprocess.h"
#include "scratch/preprocess.h"
#include "scratch/utils.h"

#include "model_base.hpp"

using namespace nvinfer1;

class ModelSegmentationScratch : public ModelBase
{
public:
    explicit ModelSegmentationScratch(const Options options) : ModelBase(options)
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
        prepare_buffer();
    }
    ~ModelSegmentationScratch() override = default;
    void infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch) override
    {
        ASSERT(false, "This method is not supported for segmentation task, please use infer with masks");
    }

    void infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch, std::vector<cv::Mat> &masks) override;


private:
    void serialize(std::string &wts_name, std::string &engine_path, std::string &type, float &gd, float &gw, int &max_channels) override;
    void deserialize(std::string &engine_path) override;
    void prepare_buffer() override;
    void parse_options(std::string &type, float &gd, float &gw, int &max_channels) override;
    std::vector<cv::Mat> process_mask(const float *proto, int proto_size, std::vector<Detection> &dets);

    Logger gLogger;
    const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
    const int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);

    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    cudaStream_t stream;

    int batch_size = 0;
    float *device_buffers[3];
    float *output_buffer_host = nullptr;
    float *output_seg_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;
    std::unordered_map<int, std::string> labels_map;
};