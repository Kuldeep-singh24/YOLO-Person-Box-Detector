from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/person_box_detector/weights/best.pt")

# Open video
cap = cv2.VideoCapture("input/video.mp4")

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output/line_crossing.mp4",
    fourcc,
    fps,
    (width, height)
)

# Horizontal virtual line
line_y = height // 2

# Store previous center position of each object
previous_positions = {}

# Store IDs that have already crossed
crossed_ids = set()

# Counters
left_to_right = 0
right_to_left = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # Track objects
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Draw virtual line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        3
    )

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, track_id, confidence in zip(
            boxes,
            class_ids,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            # Center of object
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = model.names[int(class_id)]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Display ID
            label = f"{class_name} ID:{track_id} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

            # Check whether this ID existed in previous frame
            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                # Moving from above line to below line
                if previous_y < line_y and center_y >= line_y:

                    if track_id not in crossed_ids:
                        right_to_left += 1
                        crossed_ids.add(track_id)

                # Moving from below line to above line
                elif previous_y > line_y and center_y <= line_y:

                    if track_id not in crossed_ids:
                        left_to_right += 1
                        crossed_ids.add(track_id)

            # Save current position
            previous_positions[track_id] = center_y

    # Display counters
    cv2.putText(
        frame,
        f"Left -> Right: {left_to_right}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Right -> Left: {right_to_left}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Save frame
    out.write(frame)

cap.release()
out.release()

print("Line crossing completed!")
print(f"Left -> Right: {left_to_right}")
print(f"Right -> Left: {right_to_left}")
print("Output saved to output/line_crossing.mp4")