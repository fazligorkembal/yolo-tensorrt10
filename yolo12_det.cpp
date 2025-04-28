#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"

/** Todo: Add the followings

#include "preprocess.h"
*/

#include "utils.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

int main(int argc, char **argv)
{
    // yolo12_det -s ../models/yolo12n.wts ../models/yolo12n.fp32.trt n
    // yolo12_det -d ../models/yolo12n.fp32.trt ../images c
    cudaSetDevice(kGpuId);
    std::string wts_name;
    std::string engine_name;
    std::string img_dir;
    std::string cuda_post_process;
    std::string type;
    int model_bboxes;
    float gd = 0, gw = 0;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, cuda_post_process, gd, gw, max_channels))
    {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo12_det -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolo12_det -d [.engine] ../images  [c/g]// deserialize plan file and run inference"
                  << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty())
    {
        serialize_engine(wts_name, engine_name, gd, gw, max_channels, type);
        return 0;
    }

    std::cout << "Process done" << std::endl;
    return 0;
}
