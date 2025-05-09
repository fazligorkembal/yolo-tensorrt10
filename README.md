## Introduction

Yolo11 model supports TensorRT-10.

Training code [link](https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.38.zip)

## Environment

Todo: Add Env Specs

## Support

* [x] YOLO11-pose support FP32/FP16/INT8 and Python/C++ API


## Config

* Choose the YOLO11 sub-model n/s/m/l/x from command line arguments.
* Other configs please check [include/config.h](include/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# Download ultralytics
wget https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.0.zip -O ultralytics-8.3.0.zip
# Unzip ultralytics
unzip ultralytics-8.3.0.zip
cd ultralytics-8.3.0
# Download models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O yolo11n-pose.pt
# Generate .wts
cp [PATH-TO-TENSORRTX]/yolo11/gen_wts.py .
python gen_wts.py -w yolo11n-pose.pt -o yolo11n-pose.wts -t pose
# A file 'yolo11n.wts' will be generated.
```

2. build tensorrtx/yolo11 and run
```shell
cd [PATH-TO-TENSORRTX]/yolo11
mkdir build
cd build
cmake ..
make
```

### Pose
```shell
cp [PATH-TO-ultralytics]/yolo11n-pose.wts .
# Build and serialize TensorRT engine
./yolo11_pose -s yolo11n-pose.wts yolo11n-pose.engine [n/s/m/l/x]
# Run inference
./yolo11_pose -d yolo11n-pose.engine ../images
```
