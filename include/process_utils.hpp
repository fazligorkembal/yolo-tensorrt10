#pragma once
#include <string>

enum class ModelType
{
    None,
    yolo11n_detection,
    yolo11s_detection,
    yolo11m_detection,
    yolo11l_detection,
    yolo11x_detection

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
            OptimizationType optimization_type)
        : model_path(model_path),
          labels_map(labels_map),
          model_type(model_type),
          optimization_type(optimization_type)
    {
    }

    std::string model_path = "";
    std::string labels_map = "";
    ModelType model_type = ModelType::None;
    OptimizationType optimization_type = OptimizationType::None;
};
