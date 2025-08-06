#include "process_utils.hpp"
#include "scratch/model_segmentation.hpp"
#include "scratch/model_pose.hpp"
class TensorRTForge
{
public:
    TensorRTForge(
        const std::string model_path,
        const ModelType model_type,
        const OptimizationType optimization_type,
        const std::string labels_map);
    ~TensorRTForge() = default;

    //std::unique_ptr<ModelBase> model;
    ModelBase *model = nullptr;
};
