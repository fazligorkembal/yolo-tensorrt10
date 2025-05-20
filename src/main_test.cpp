#include "scratch/model_segmentation.hpp"

int main(int argc, char **argv)
{
    std::string model_path = "/home/user/Documents/tensorrtforge/build/yolo11n-seg.wts";
    std::string labels_map = "/home/user/Documents/tensorrtforge/build/coco.txt";
    ModelType model_type = ModelType::yolo11n;
    ConversionType conversion_type = ConversionType::scratch;
    TaskType task_type = TaskType::segmentation;
    OptimizationType optimization_type = OptimizationType::fp32;

    Options options(model_path, labels_map, model_type, conversion_type, task_type, optimization_type);
    ModelSegmentationScratch model(options);
    

    std::string input_path = "/home/user/Documents/tensorrtforge/input_samples/test_input.jpg";
    cv::Mat img = cv::imread(input_path);
    cv::resize(img, img, cv::Size(640, 640));
    std::vector<cv::Mat> images;
    images.push_back(img);

    std::vector<std::vector<Detection>> res_batch;
    std::vector<cv::Mat> masks;

    model.infer(images, res_batch, masks);

    cv::imshow("img", images[0]);
    cv::waitKey(0);

    return 0;
    
}