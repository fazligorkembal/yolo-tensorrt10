#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &img_dir, std::string &type,
                std::string &cuda_post_process, float &gd, float &gw, int &max_channels)
{
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7))
    {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto sub_type = std::string(argv[4]);
        if (sub_type[0] == 'n')
        {
            gd = 0.50;
            gw = 0.25;
            max_channels = 1024;
            type = "n";
        }
        else if (sub_type[0] == 's')
        {
            gd = 0.50;
            gw = 0.50;
            max_channels = 1024;
            type = "s";
        }
        else if (sub_type[0] == 'm')
        {
            gd = 0.50;
            gw = 1.00;
            max_channels = 512;
            type = "m";
        }
        else if (sub_type[0] == 'l')
        {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
            type = "l";
        }
        else if (sub_type[0] == 'x')
        {
            gd = 1.0;
            gw = 1.50;
            max_channels = 512;
            type = "x";
        }
        else
        {
            return false;
        }
    }
    else if (std::string(argv[1]) == "-d" && argc == 5)
    {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
    }
    else
    {
        return false;
    }
    return true;
}

void serialize_engine(std::string &wts_name, std::string &engine_name, std::string &type, float &gd, float &gw,
                      int &max_channels)
{
    std::cout << "Building Yolo12 engine..." << std::endl;
}

int main(int argc, char **argv)
{
    // yolo11_pose -s ../models/yolo11n-pose.wts ../models/yolo11n-pose.fp32.trt n
    // yolo11_pose -d ../models/yolo11n-pose.fp32.trt ../images c
    cudaSetDevice(kGpuId);
    std::string wts_name;
    std::string engine_name;
    std::string img_dir;
    std::string type;
    std::string cuda_post_process;
    int model_bboxes;
    float gd = 0.0f, gw = 0.0f;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, cuda_post_process, gd, gw, max_channels))
    {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo11_pose -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolo11_pose -d [.engine] ../images  [c/g]// deserialize plan file and run inference"
                  << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty())
    {
        serialize_engine(wts_name, engine_name, type, gd, gw, max_channels);
        return 0;
    }

    std::cout << "Process Done!" << std::endl;
}