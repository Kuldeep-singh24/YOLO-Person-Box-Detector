import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import subprocess
import imageio_ffmpeg


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YOLO Person & Box Detector",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🔍 YOLO Person & Box Detector")

st.write(
    "AI-powered Person and Box detection using a custom-trained YOLO model."
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = YOLO("models/best.pt")


# =========================================================
# SELECT INPUT TYPE
# =========================================================

input_type = st.radio(
    "Choose Input Type",
    ["Image", "Video"],
    horizontal=True
)


# =========================================================
# IMAGE DETECTION
# =========================================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")

        st.image(
            image,
            width="stretch"
        )

        # Run YOLO
        results = model.predict(
            image,
            conf=0.25
        )

        # Detection result
        result_image = results[0].plot()

        st.subheader("Detection Result")

        st.image(
            result_image,
            channels="BGR",
            width="stretch"
        )

        # =================================================
        # DETECTION SUMMARY
        # =================================================

        st.subheader("Detection Summary")

        person_count = 0
        box_count = 0

        if results[0].boxes is not None:

            class_ids = results[0].boxes.cls.tolist()

            for class_id in class_ids:

                class_name = model.names[int(class_id)]

                if class_name == "person":
                    person_count += 1

                elif class_name == "box":
                    box_count += 1

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Persons",
                person_count
            )

        with col2:
            st.metric(
                "Boxes",
                box_count
            )


# =========================================================
# VIDEO DETECTION + TRACKING
# =========================================================

else:

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # =================================================
        # SAVE UPLOADED VIDEO
        # =================================================

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(
            uploaded_video.read()
        )

        temp_input.close()

        st.subheader("Processing Video...")

        # =================================================
        # YOLO TRACKING
        # =================================================

        results = model.track(
            source=temp_input.name,
            conf=0.25,
            tracker="bytetrack.yaml",
            save=True,
            persist=True
        )

        st.success(
            "Video processing completed!"
        )

        # =================================================
        # FIND YOLO OUTPUT VIDEO
        # =================================================

        result_dir = str(results[0].save_dir)

        video_files = []

        for file_name in os.listdir(result_dir):

            if file_name.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):

                video_files.append(
                    os.path.join(
                        result_dir,
                        file_name
                    )
                )

        if video_files:

            input_processed_video = video_files[0]

            # =================================================
            # CONVERT TO BROWSER-FRIENDLY H.264 MP4
            # =================================================

            browser_video = os.path.join(
                tempfile.gettempdir(),
                "yolo_processed_h264.mp4"
            )

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg_path,
                "-y",
                "-i",
                input_processed_video,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                browser_video
            ]

            with st.spinner(
                "Converting video for browser playback..."
            ):

                subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )

            # =================================================
            # DISPLAY PROCESSED VIDEO
            # =================================================

            st.subheader(
                "Processed Video"
            )

            with open(
                browser_video,
                "rb"
            ) as video_file:

                video_bytes = video_file.read()

            st.video(
                video_bytes,
                format="video/mp4"
            )

            st.success(
                "YOLO detection and object tracking "
                "are visible in the processed video."
            )

        else:

            st.error(
                "Processed video could not be found."
            )

        # =================================================
        # CLEAN TEMPORARY INPUT
        # =================================================

        os.unlink(
            temp_input.name
        )