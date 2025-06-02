#include <filesystem>
#include "tracker/trackerlib.hpp"
#include "string"


Tracker::Tracker()
{
    //todo: fix python path problem ... !
    pybind11::module_::import("sys").attr("path").attr("append")(std::string(MODULE_PATH));
    tracker_module = py::module_::import("tracker_compute");
    deep_sort_compute = tracker_module.attr("deep_sort_compute");

    detection_track_.size = 0;
    detection_track_.bbox = new float[400];
    detection_track_.class_id = new float[100];
    detection_track_.conf = new float[100];
    

}



void Tracker::track_from_scratch(cv::Mat &image, std::vector<std::vector<Detection>> &detections)
{
    img_array = py::array_t<uint8_t>(
        {image.rows, image.cols, image.channels()},
        {static_cast<size_t>(image.step[0]), static_cast<size_t>(image.step[1]), static_cast<size_t>(image.elemSize1())},
        image.data);

    detection_track_.size = detections[0].size();
    for (size_t i = 0; i < detections[0].size(); ++i) {
        detection_track_.bbox[i * 4] = static_cast<float>(detections[0][i].bbox[0]);
        detection_track_.bbox[i * 4 + 1] = static_cast<float>(detections[0][i].bbox[1]);
        detection_track_.bbox[i * 4 + 2] = static_cast<float>(detections[0][i].bbox[2]);
        detection_track_.bbox[i * 4 + 3] = static_cast<float>(detections[0][i].bbox[3]);

        detection_track_.class_id[i] = detections[0][i].class_id;
        detection_track_.conf[i] = detections[0][i].conf;

        

    }
    

    deep_sort_compute(img_array, detection_track_);
}