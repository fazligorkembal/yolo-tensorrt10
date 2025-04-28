import onnx
import cv2

import onnxruntime as ort
import numpy as np
from onnx import numpy_helper

from ultralytics import YOLO

#model = YOLO("yolo12n.pt")
#model.export(format="onnx")


def is_same(data1, data2, eps=0.01):
    if len(data1) != len(data2):
        print("Length mismatch", len(data1), len(data2))
        return False
    else:
        for d1, d2 in zip(data1, data2):
            if abs(d1 - d2) > eps:
                return False
    
    return True

model = onnx.load("/home/user/Documents/tensorrtx/yolo12n.onnx")

def read_txt_file(file_path):
    data = []
    with open(file_path, "rb") as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line))
    return data

initializers = {init.name: init for init in model.graph.initializer}

hold_outputs = []
for i in model.graph.node:
    #print("Node name:", i.name, "Node op_type:", i.op_type, "Node input:", i.input, "Node output:", i.output)
    print("Node name:", i.name)
    if "/model.6/cv2/act/Mul" == i.name:
        for o in i.output:
            new_output = onnx.ValueInfoProto()
            new_output.name = o
            hold_outputs.append(new_output)


model.graph.output.extend(hold_outputs)
onnx.save(model, "/home/user/Documents/tensorrtx/yolo12n_cropped.onnx")

image = np.zeros((640, 640, 3), dtype=np.float32)

image = image.transpose(2, 0, 1)
image = image.reshape(1, 3, 640, 640)
image = image.astype("float32") / 255.0

ort_session = ort.InferenceSession("/home/user/Documents/tensorrtx/yolo12n_cropped.onnx")
outputs = ort_session.run(None, {"images": image})

print(len(outputs))
print(type(outputs))
print(type(outputs[0]))
print(outputs[0].shape)
print(outputs[1].shape)

flatten_output = outputs[1].flatten()
flatten_output_trt = read_txt_file("/home/user/Documents/tensorrtx/tensorrtx/yolov12/build/results.txt")

res = is_same(flatten_output, flatten_output_trt, 0.01)

errors = []

if res:
    print("Results are same")
else:
    for i, (o, t) in enumerate(zip(flatten_output, flatten_output_trt)):
        if abs(o - t) > 0.01:
            #print("Mismatch at index:", i, "Output:", o, "TRT output:", t, "Ratio:", abs(o/t))
            errors.append(abs(o - t))
        
        
    print("Max error: ", max(errors))
    print("Out of error count: ", len(errors))
    print("Length of outputs: ", len(flatten_output), len(flatten_output_trt))