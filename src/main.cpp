#include "tensorrtforge.hpp"

int main(int argc, char** argv)
{
    std::string model_path = "path/to/your/model.trt"; // Replace with your actual model path
    TensorRTForge forge(model_path, ModelType::yolo11n_det, OptimizationType::fp16, "");

}