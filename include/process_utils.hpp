#pragma once
#include <string>

enum class ModelName
{
    None,
    yolo11,

};

enum class ModelSize
{
    None,
    n,
    s,
    m,
    l,
    x
};

enum class TaskType
{
    None,
    Detection,
};

enum class OptimizationType
{
    None,
    fp16,
    int8,
    fp32
};

class Options
{
public:
    Options(std::string model_path,
            std::string labels_map,
            ModelName model_name,
            ModelSize model_size,
            TaskType task_type = TaskType::None,
            OptimizationType optimization_type = OptimizationType::fp32)
        : model_path(model_path),
          labels_map(labels_map),
          model_name(model_name),
          model_size(model_size),
          task_type(task_type),
          optimization_type(optimization_type)
    {
    }

    std::string model_path = "";
    std::string labels_map = "";
    ModelName model_name = ModelName::None;
    ModelSize model_size = ModelSize::None;
    TaskType task_type = TaskType::None;
    OptimizationType optimization_type = OptimizationType::None;
};
