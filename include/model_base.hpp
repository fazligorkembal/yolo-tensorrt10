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
        ASSERT(options_.model_type != ModelType::None, "Model type cannot be None, please check available model types in include/utils.hpp");
        ASSERT(options_.conversion_type != ConversionType::None, "Conversion type cannot be None, please check available conversion types in include/utils.hpp");
        ASSERT(options_.task_type != TaskType::None, "Task type cannot be None, please check available task types in include/utils.hpp");
        ASSERT(options_.optimization_type != OptimizationType::None, "Optimization type cannot be None, please check available optimization types in include/utils.hpp");

        ASSERT(options_.model_path.ends_with(".wts") || options_.model_path.ends_with(".engine") ||
                   options_.model_path.ends_with(".plan"),
               "Model path must be a .wts, .engine or .plan file for scratch conversion");
    }

    virtual ~ModelBase() = default;

protected:
    Options options_;
    virtual void serialize(std::string &wts_name, std::string &engine_path, std::string &type, float &gd, float &gw, int &max_channels) = 0;
    virtual void deserialize(std::string &engine_path) = 0;
    virtual void prepare_buffer(TaskType task_type) = 0;
    virtual void parse_options(std::string &type, float &gd, float &gw, int &max_channels) = 0;

    int kGpuId = 0;

};
