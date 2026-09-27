from ultralytics import YOLO

# Load our trained YOLO model
model = YOLO("runs/detect/person_box_detector/weights/best.pt")

# Run YOLO detection on the video
model.predict(
    source="input/video.mp4",
    conf=0.25,
    save=True,
    project="output",
    name="video_detection"
)

print("Video detection completed!")
print("Output saved in output/video_detection/")