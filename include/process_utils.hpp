#pragma once
#include <string>

enum class ModelType
{
    None,
    yolo11n,
    yolo11s,
    yolo11m,
    yolo11l,
    yolo11x

};

enum class TaskType
{
    None,
    detection,
    segmentation,
    classification,
    pose,
    obb
};

enum class ConversionType
{
    None,
    scratch,
    automatic
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
            ModelType model_type,
            ConversionType conversion_type,
            TaskType task_type,
            OptimizationType optimization_type)
        : model_path(model_path),
          labels_map(labels_map),
          model_type(model_type),
          conversion_type(conversion_type),
          task_type(task_type),
          optimization_type(optimization_type)
    {
    }

    std::string model_path = "";
    std::string labels_map = "";
    ModelType model_type = ModelType::None;
    ConversionType conversion_type = ConversionType::None;
    TaskType task_type = TaskType::None;
    OptimizationType optimization_type = OptimizationType::None;
};
