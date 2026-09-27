from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("runs/detect/person_box_detector/weights/best.pt")

# Open input video
cap = cv2.VideoCapture("input/video.mp4")

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output/object_counting.mp4",
    fourcc,
    fps,
    (width, height)
)

# Keep track of unique IDs
person_ids = set()
box_ids = set()

while True:
    success, frame = cap.read()

    if not success:
        break

    # Track objects in current frame
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, track_id, confidence in zip(
            boxes, class_ids, track_ids, confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            class_name = model.names[int(class_id)]

            # Store unique IDs
            if class_name == "person":
                person_ids.add(track_id)

            elif class_name == "box":
                box_ids.add(track_id)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Label
            label = f"{class_name} ID:{track_id} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # Display counts
    cv2.putText(
        frame,
        f"Persons: {len(person_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Boxes: {len(box_ids)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Save frame
    out.write(frame)

cap.release()
out.release()

print("Object counting completed!")
print(f"Unique persons detected: {len(person_ids)}")
print(f"Unique boxes detected: {len(box_ids)}")
print("Output saved to output/object_counting.mp4")