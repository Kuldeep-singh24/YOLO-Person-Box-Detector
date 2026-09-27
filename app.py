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

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


model = load_model()


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

        # Fixed: no width="stretch"
        st.image(image)

        # =================================================
        # RUN YOLO
        # =================================================

        results = model.predict(
            image,
            conf=0.25
        )

        # =================================================
        # DETECTION RESULT
        # =================================================

        result_image = results[0].plot()

        st.subheader("Detection Result")

        st.image(
            result_image,
            channels="BGR"
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

        try:

            results = model.track(
                source=temp_input.name,
                conf=0.25,
                tracker="bytetrack.yaml",
                save=True,
                persist=True
            )

        except Exception as e:

            st.error("Video tracking failed.")

            st.exception(e)

            try:
                os.unlink(temp_input.name)
            except Exception:
                pass

            st.stop()

        st.success(
            "YOLO detection and tracking completed!"
        )

        # =================================================
        # FIND YOLO OUTPUT VIDEO
        # =================================================

        result_dir = str(results[0].save_dir)

        video_files = []

        if os.path.exists(result_dir):

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

        # =================================================
        # CHECK OUTPUT
        # =================================================

        if not video_files:

            st.error(
                "YOLO finished processing, but the output video "
                "could not be found."
            )

            try:
                os.unlink(temp_input.name)
            except Exception:
                pass

            st.stop()

        input_processed_video = video_files[0]

        st.info(
            f"YOLO output found: "
            f"{os.path.basename(input_processed_video)}"
        )

        # =================================================
        # CONVERT VIDEO TO H.264 MP4
        # =================================================

        browser_video = os.path.join(
            tempfile.gettempdir(),
            "yolo_processed_h264.mp4"
        )

        try:

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

                process = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            # =================================================
            # CHECK FFMPEG RESULT
            # =================================================

            if process.returncode != 0:

                st.error(
                    "Video conversion failed."
                )

                st.code(
                    process.stderr[-4000:]
                )

                st.info(
                    "YOLO tracking itself completed successfully."
                )

            elif not os.path.exists(browser_video):

                st.error(
                    "Converted video file was not created."
                )

            else:

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
                    "Processed video displayed successfully "
                    "with YOLO tracking IDs."
                )

        except Exception as e:

            st.error(
                "An error occurred while preparing the video."
            )

            st.exception(e)

        # =================================================
        # CLEAN TEMPORARY INPUT
        # =================================================

        try:

            os.unlink(
                temp_input.name
            )

        except Exception:
            pass