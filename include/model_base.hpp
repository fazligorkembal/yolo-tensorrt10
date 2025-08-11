#pragma once

#include <fstream>
#include <filesystem>

#include "assert.hpp"
#include "process_utils.hpp"


class ModelBase
{
public:
    explicit ModelBase(const Options options) : options_(options)
    {
        ASSERT(!options_.model_path.empty(), "Model path cannot be empty");
        ASSERT(std::filesystem::exists(options_.model_path) && std::filesystem::is_regular_file(options_.model_path), "Model path must be full-path and a valid file");
        ASSERT(options_.model_name != ModelName::None, "Model name cannot be None, please check available model names in include/process_utils.hpp");
        ASSERT(options_.task_type != TaskType::None, "Task type cannot be None, please check available task types in include/utils.hpp");
        ASSERT(options_.optimization_type != OptimizationType::None, "Optimization type cannot be None, please check available optimization types in include/utils.hpp");

        ASSERT(options_.model_path.ends_with(".wts") || options_.model_path.ends_with(".engine") ||
                   options_.model_path.ends_with(".plan"),
               "Model path must be a .wts, .engine or .plan file for scratch conversion");
    }

    virtual void infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch) = 0;
    virtual void infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch, std::vector<cv::Mat> &masks) = 0;

    virtual ~ModelBase() = default;

protected:
    Options options_;
    virtual void serialize(std::string &wts_name, std::string &engine_path, std::string &type, float &gd, float &gw, int &max_channels) = 0;
    virtual void deserialize(std::string &engine_path) = 0;
    virtual void parse_options(std::string &type, float &gd, float &gw, int &max_channels) = 0;
    virtual void prepare_buffer() = 0;
    int kGpuId = 0;

};
