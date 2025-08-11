from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # Load a pre-trained YOLOv8n model
model.export(format="onnx")  # Export to