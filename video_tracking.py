from ultralytics import YOLO

# Load our trained model
model = YOLO("runs/detect/person_box_detector/weights/best.pt")

# Run object tracking on the video
results = model.track(
    source="input/video.mp4",
    conf=0.25,
    save=True,
    tracker="bytetrack.yaml",
    project="output",
    name="video_tracking",
    persist=True
)

print("Object tracking completed!")
print("Output saved in output/video_tracking/")