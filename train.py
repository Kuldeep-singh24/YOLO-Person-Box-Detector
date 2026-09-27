from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train the model on our custom dataset
results = model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="person_box_detector"
)

print("Training completed!")