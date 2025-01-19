from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
