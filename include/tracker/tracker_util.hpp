#pragma once

struct DetectionTrack
{
    size_t size;
    float *bbox;
    float *class_id;
    float *conf;
};

