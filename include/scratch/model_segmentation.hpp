#pragma once



#include "model.hpp"

class ModelSegmentationScratch : public Model
{
public:
    explicit ModelSegmentationScratch(const Options options) : Model(options)
    {
        ASSERT(options_.model_path.ends_with(".wts") || options_.model_path.ends_with(".engine") ||
                   options_.model_path.ends_with(".plan"),
               "Model path must be a .wts, .engine or .plan file for scratch conversion");
        //todo: add int8 support
        ASSERT(options_.optimization_type == OptimizationType::fp32 || 
                   options_.optimization_type == OptimizationType::fp16, "Optimization type must be fp32 or fp16 for scratch conversion, int8 is not supported yet");
        
        std::string wts_path = options_.model_path;
        std::string engine_path = wts_path.substr(0, wts_path.find_last_of('.')) + ".engine";
        
    }
    ~ModelSegmentationScratch() override = default;

    void serialize() override;

    void deserialize() override;
};