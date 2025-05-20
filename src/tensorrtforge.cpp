#include <iostream>
#include <filesystem>

#include "assert.hpp"
#include "tensorrtforge.hpp"


TensorRTForge::TensorRTForge(const std::string model_path,
           const ModelType model_type,
           const ConversionType conversion_type,
           const TaskType task_type,
           const OptimizationType optimization_type,
           const std::string labels_map)
{
    ASSERT(!model_path.empty(), "Model path cannot be empty");
    ASSERT(std::filesystem::exists(model_path) && std::filesystem::is_regular_file(model_path), "Model path must be full-path and a valid file");
    ASSERT(model_type != ModelType::None, "Model type cannot be None, please check available model types in include/process_utils.hpp");
    ASSERT(conversion_type != ConversionType::None, "Conversion type cannot be None, please check available conversion types in include/process_utils.hpp");
    ASSERT(task_type != TaskType::None, "Task type cannot be None, please check available task types in include/process_utils.hpp");
    ASSERT(optimization_type != OptimizationType::None, "Optimization type cannot be None, please check available optimization types in include/process_utils.hpp");
    ASSERT(model_path.ends_with(".wts") || model_path.ends_with(".engine") ||
               model_path.ends_with(".plan"),
           "Model path must be a .wts, .engine or .plan file for scratch conversion");
    if(task_type == TaskType::segmentation)
    {
        ASSERT(!labels_map.empty(), "Labels map cannot be empty");
        ASSERT(std::filesystem::exists(labels_map) && std::filesystem::is_regular_file(labels_map), "Labels map must be full-path and a valid file");
    }


    switch (task_type)
    {
        case TaskType::segmentation:
            //model = std::make_unique<ModelSegmentationScratch>(Options(model_path, labels_map, model_type, conversion_type, task_type, optimization_type));
            model = new ModelSegmentationScratch(Options(model_path, labels_map, model_type, conversion_type, task_type, optimization_type));
            break;
        case TaskType::detection:
            // model_ = std::make_unique<ModelDetectionScratch>(Options(model_path, model_type, conversion_type, task_type, optimization_type));
            break;
        default:
            ASSERT(false, "Task type not supported");
            break;
    }

}
