from ultralytics import YOLO

# Load our trained model
model = YOLO("runs/detect/person_box_detector/weights/best.pt")

# Evaluate on the test dataset
results = model.val(
    data="dataset/data.yaml",
    split="test",
    imgsz=640
)

print("\n===== TEST SET EVALUATION =====")
print(f"mAP50:     {results.box.map50:.3f}")
print(f"mAP50-95:  {results.box.map:.3f}")
print(f"Precision: {results.box.mp:.3f}")
print(f"Recall:    {results.box.mr:.3f}")