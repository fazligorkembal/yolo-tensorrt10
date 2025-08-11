#include <iostream>
#include <filesystem>

#include "assert.hpp"
#include "tensorrtforge.hpp"


TensorRTForge::TensorRTForge(const std::string model_path,
           const ModelName model_name,
           const ModelSize model_size,
           const TaskType task_type,
           const OptimizationType optimization_type,
           const std::string labels_map)
{
    ASSERT(!model_path.empty(), "Model path cannot be empty");
    ASSERT(std::filesystem::exists(model_path) && std::filesystem::is_regular_file(model_path), "Model path must be full-path and a valid file");
    ASSERT(model_name != ModelName::None, "Model name cannot be None, please check available model names in include/process_utils.hpp");
    ASSERT(task_type != TaskType::None, "Task type cannot be None, please check available task types in include/process_utils.hpp");
    ASSERT(optimization_type != OptimizationType::None, "Optimization type cannot be None, please check available optimization types in include/process_utils.hpp");
    ASSERT(model_path.ends_with(".wts") || model_path.ends_with(".engine") ||
               model_path.ends_with(".plan"),
           "Model path must be a .wts, .engine or .plan file for scratch conversion");
    
    
    if(task_type == TaskType::Detection)
    {
        model = new ModelDetectionScratch(Options(model_path, labels_map, model_name, model_size, task_type, optimization_type));
    }
    else
    {
        ASSERT(false, "Unsupported task type");
    }


}
