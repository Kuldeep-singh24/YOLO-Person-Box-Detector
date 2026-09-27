from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/person_box_detector/weights/best.pt")

# Test images
results = model.predict(
    source="input",
    conf=0.25,
    save=True
)

print("Prediction completed!")
print("Results saved in the runs/detect folder.")