#pragma once

#include <fstream>

#include "assert.hpp"
#include "process_utils.hpp"

class Model
{
    explicit Model(const Options options) : options_(options)
    {
        ASSERT(!options_.model_path.empty(), "Model path cannot be empty");
        ASSERT(std::filesystem::exists(options_.model_path) && std::filesystem::is_regular_file(options_.model_path), "Model path must be full-path and a valid file");
        ASSERT(options_.model_type != ModelType::None, "Model type cannot be None, please check available model types in include/utils.hpp");
        ASSERT(options_.conversion_type != ConversionType::None, "Conversion type cannot be None, please check available conversion types in include/utils.hpp");
        ASSERT(options_.task_type != TaskType::None, "Task type cannot be None, please check available task types in include/utils.hpp");
        ASSERT(options_.optimization_type != OptimizationType::None, "Optimization type cannot be None, please check available optimization types in include/utils.hpp");
        ASSERT(options_.input_type != InputType::None, "Input type cannot be None, please check available input types in include/utils.hpp");
    }

    virtual ~Model() = default;

    virtual void serialize() = 0;
    virtual void deserialize() = 0;
protected:
    Options options_;
}