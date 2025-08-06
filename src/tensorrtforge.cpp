#include <iostream>
#include <filesystem>

#include "assert.hpp"
#include "tensorrtforge.hpp"


TensorRTForge::TensorRTForge(const std::string model_path,
           const ModelType model_type,
           const OptimizationType optimization_type,
           const std::string labels_map)
{
    ASSERT(!model_path.empty(), "Model path cannot be empty");
    ASSERT(std::filesystem::exists(model_path) && std::filesystem::is_regular_file(model_path), "Model path must be full-path and a valid file");
    ASSERT(model_type != ModelType::None, "Model type cannot be None, please check available model types in include/process_utils.hpp");
    ASSERT(optimization_type != OptimizationType::None, "Optimization type cannot be None, please check available optimization types in include/process_utils.hpp");
    ASSERT(model_path.ends_with(".wts") || model_path.ends_with(".engine") ||
               model_path.ends_with(".plan"),
           "Model path must be a .wts, .engine or .plan file for scratch conversion");
    



}
