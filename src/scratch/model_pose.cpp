#include "scratch/model_pose.hpp"
#include "scratch/model_builder.h"

void ModelPoseScratch::serialize(std::string &wts_name, std::string &engine_path, std::string &type, float &gd, float &gw, int &max_channels)
{
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    serialized_engine = buildEngineYolo11Pose(builder, config, DataType::kFLOAT, options_.model_path, gd, gw, max_channels, type);

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

void ModelPoseScratch::deserialize(std::string &engine_path)
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

    // prepare_buffer();

    std::cout << "Output dimensions: " << out_dims.d[0] << ", " << out_dims.d[1] << ", " << out_dims.d[2] << ", " << out_dims.d[3] << std::endl;
    std::cout << "Engine deserialized successfully, " << engine_path << std::endl;
}

void ModelPoseScratch::prepare_buffer()
{
    ASSERT(engine->getNbIOTensors() == 2, "Engine must have 2 IO tensors but got " + std::to_string(engine->getNbIOTensors()));

    CUDA_CHECK(cudaMalloc((void **)&device_buffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&device_buffers[1], kBatchSize * kOutputSize * sizeof(float)));

    context->setTensorAddress(kInputTensorName, (float *)device_buffers[0]);
    context->setTensorAddress(kOutputTensorName, (float *)device_buffers[1]);

    output_buffer_host = new float[kBatchSize * kOutputSize];
}

void ModelPoseScratch::infer(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &res_batch)
{
    
    cuda_batch_preprocess(images, device_buffers[0], kInputW, kInputH, stream);

    context->enqueueV3(stream);

    // todo: add gpu post process here
    CUDA_CHECK(cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // todo: add gpu post process here
    batch_nms(res_batch, output_buffer_host, images.size(), kOutputSize, kConfThresh, kNmsThresh);

    draw_bbox_keypoints_line(images, res_batch);
}

void ModelPoseScratch::parse_options(std::string &type, float &gd, float &gw, int &max_channels)
{
    
}