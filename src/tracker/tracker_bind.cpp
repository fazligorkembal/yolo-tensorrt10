#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tracker/tracker_util.hpp"

namespace py = pybind11;



PYBIND11_MODULE(pybind_tracker, m)
{
    m.doc() = "pybind11 tracker plugin"; // Optional module docstring

    py::class_<DetectionTrack>(m, "Detection")
        .def(py::init<>())
        .def_readonly("size", &DetectionTrack::size)
        .def_property_readonly("bbox_array", [](const DetectionTrack &d)
                               { return py::array_t<float>(
                                     {d.size * 4},
                                     {sizeof(float)},
                                     d.bbox,
                                     py::cast(d)); })
        .def_property_readonly("class_id_array", [](const DetectionTrack &d)
                               { return py::array_t<float>(
                                     {d.size},
                                     {sizeof(float)},
                                     d.class_id,
                                     py::cast(d)); })
        .def_property_readonly("conf_array", [](const DetectionTrack &d)
                               { return py::array_t<float>(
                                     {d.size},
                                     {sizeof(float)},
                                     d.conf,
                                     py::cast(d)); });
}