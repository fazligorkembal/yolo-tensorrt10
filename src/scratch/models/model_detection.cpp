#include "scratch/models/model_detection.hpp"
#include "scratch/model_builder.h"

void ModelDetectionScratch::serialize(std::string &wts_name, std::string &engine_path, std::string &type, float &gd, float &gw, int &max_channels)
{
}

void ModelDetectionScratch::deserialize(std::string &engine_path)
{

}

void ModelDetectionScratch::prepare_buffer(TaskType task_type)
{

}

void ModelDetectionScratch::infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch)
{

}

void ModelDetectionScratch::parse_options(std::string &type, float &gd, float &gw, int &max_channels)
{
    if (options_.model_type == ModelType::yolo11n)
    {
        gd = 0.50;
        gw = 0.25;
        max_channels = 1024;
        type = "n";
    }
    else if (options_.model_type == ModelType::yolo11s)
    {
        gd = 0.50;
        gw = 0.50;
        max_channels = 1024;
        type = "s";
    }
    else if (options_.model_type == ModelType::yolo11m)
    {
        gd = 0.50;
        gw = 1.00;
        max_channels = 512;
        type = "m";
    }
    else if (options_.model_type == ModelType::yolo11l)
    {
        gd = 1.0;
        gw = 1.0;
        max_channels = 512;
        type = "l";
    }
    else if (options_.model_type == ModelType::yolo11x)
    {
        gd = 1.0;
        gw = 1.50;
        max_channels = 512;
        type = "x";
    }
    else
    {
        ASSERT(false, "Invalid model type");
    }

    ASSERT(gd > 0 && gw > 0, "gd and gw must be greater than 0");
    ASSERT(max_channels > 0, "max_channels must be greater than 0");
    ASSERT(type == "n" || type == "s" || type == "m" || type == "l" || type == "x", "Invalid model type");
}