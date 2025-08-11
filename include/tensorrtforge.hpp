#include "process_utils.hpp"
#include "scratch/model_segmentation.hpp"
#include "scratch/model_pose.hpp"
#include "scratch/model_detection.hpp"

class TensorRTForge
{
public:
    TensorRTForge(
        const std::string model_path,
        const ModelName model_name,
        const ModelSize model_size,
        const TaskType task_type,
        const OptimizationType optimization_type,

        const std::string labels_map);
    ~TensorRTForge() = default;

    //std::unique_ptr<ModelBase> model;
    ModelBase *model = nullptr;
};
