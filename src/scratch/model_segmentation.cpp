#include "scratch/model_segmentation.hpp"
#include "scratch/model_builder.h"

void ModelSegmentationScratch::serialize(std::string &wts_name, std::string &engine_path, std::string &type, float &gd, float &gw, int &max_channels)
{
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    if (options_.model_type == ModelType::yolo11n || options_.model_type == ModelType::yolo11x)
    {
        serialized_engine = buildEngineYolo11Seg(builder, config, DataType::kFLOAT, options_.model_path, gd, gw, max_channels, type);
    }
    else
    {
        ASSERT(false, "This model type is not supported yet, please check the model types for supported models");
    }

    ASSERT(serialized_engine != nullptr, "Failed to build engine");
    std::ofstream p(engine_path, std::ios::binary);
    if (!p)
    {
        ASSERT(false, "Failed to open plan output file");
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete serialized_engine;
    delete config;
    delete builder;
}

void ModelSegmentationScratch::deserialize(std::string &engine_path)
{
    ASSERT(std::filesystem::exists(engine_path) && std::filesystem::is_regular_file(engine_path), "Engine path must be full-path and a valid file");
    std::ifstream file(engine_path, std::ios::binary);

    if (!file.good())
    {
        ASSERT(false, "Failed to open engine file");
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    ASSERT(runtime != nullptr, "Failed to create runtime");
    engine = runtime->deserializeCudaEngine(serialized_engine, size);
    ASSERT(engine != nullptr, "Failed to create engine");
    context = engine->createExecutionContext();
    ASSERT(context != nullptr, "Failed to create context");

    delete[] serialized_engine;

    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getTensorShape(kOutputTensorName);
    batch_size = out_dims.d[0];

    // prepare_buffer(options_.task_type);

    std::cout << "Output dimensions: " << out_dims.d[0] << ", " << out_dims.d[1] << ", " << out_dims.d[2] << ", " << out_dims.d[3] << std::endl;
    std::cout << "Engine deserialized successfully, " << engine_path << std::endl;
}

void ModelSegmentationScratch::parse_options(std::string &type, float &gd, float &gw, int &max_channels)
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

void ModelSegmentationScratch::prepare_buffer(TaskType task_type)
{
    ASSERT(task_type == TaskType::segmentation, "Task type must be segmentation");

    ASSERT(engine->getNbIOTensors() == 3, "Engine must have 3 IO tensors but got " << engine->getNbIOTensors());

    CUDA_CHECK(cudaMalloc((void **)&device_buffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&device_buffers[1], kBatchSize * kOutputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&device_buffers[2], kBatchSize * kOutputSegSize * sizeof(float)));

    context->setTensorAddress(kInputTensorName, (float *)device_buffers[0]);
    context->setTensorAddress(kOutputTensorName, (float *)device_buffers[1]);
    context->setTensorAddress(kProtoTensorName, (float *)device_buffers[2]);

    output_buffer_host = new float[kBatchSize * kOutputSize];
    output_seg_buffer_host = new float[kBatchSize * kOutputSegSize];

    // todo: change this with dynamic label txt
    read_labels(options_.labels_map, labels_map);
    ASSERT(kNumClass == labels_map.size(), "Number of classes must be equal to labels map size, " << kNumClass << " != " << labels_map.size());
}

void ModelSegmentationScratch::infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch, std::vector<cv::Mat> &masks)
{
    cuda_batch_preprocess(images, device_buffers[0], kInputW, kInputH, stream);

    context->enqueueV3(stream);

    // todo: add gpu post process here
    CUDA_CHECK(cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output_seg_buffer_host, device_buffers[2], kBatchSize * kOutputSegSize * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // todo: add gpu post process here
    batch_nms(res_batch, output_buffer_host, images.size(), kOutputSize, kConfThresh, kNmsThresh);
    
    for (size_t i = 0; i < images.size(); i++)
    {
        auto &det = res_batch[i];
        cv::Mat img = images[i];
        auto masks_temp = process_mask(&output_seg_buffer_host[i * kOutputSegSize], kOutputSegSize, det);
        draw_mask_bbox(img, det, masks_temp, labels_map);
    }
        
}

static cv::Rect get_downscale_rect(float bbox[4], float scale)
{

    float left = bbox[0];
    float top = bbox[1];
    float right = bbox[0] + bbox[2];
    float bottom = bbox[1] + bbox[3];

    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    right = right > kInputW ? kInputW : right;
    bottom = bottom > kInputH ? kInputH : bottom;

    left /= scale;
    top /= scale;
    right /= scale;
    bottom /= scale;
    return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
}

std::vector<cv::Mat> ModelSegmentationScratch::process_mask(const float *proto, int proto_size, std::vector<Detection> &dets)
{

    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++)
    {

        cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
        auto r = get_downscale_rect(dets[i].bbox, 4);

        for (int x = r.x; x < r.x + r.width; x++)
        {
            for (int y = r.y; y < r.y + r.height; y++)
            {
                float e = 0.0f;
                for (int j = 0; j < 32; j++)
                {
                    e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask_mat.at<float>(y, x) = e;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
        masks.push_back(mask_mat);
    }
    return masks;
}
