import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import subprocess
import imageio_ffmpeg
import cv2


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

        output_video = os.path.join(
            tempfile.gettempdir(),
            "yolo_tracking_output.mp4"
        )

        # Remove previous output if it exists
        if os.path.exists(output_video):
            try:
                os.remove(output_video)
            except Exception:
                pass

        # =================================================
        # OPEN INPUT VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            temp_input.name
        )

        if not cap.isOpened():

            st.error(
                "Could not open uploaded video."
            )

            try:
                os.unlink(temp_input.name)
            except Exception:
                pass

            st.stop()

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        cap.release()

        if fps <= 0:
            fps = 30

        # =================================================
        # VIDEO WRITER
        # =================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            st.error(
                "Could not create output video."
            )

            try:
                os.unlink(temp_input.name)
            except Exception:
                pass

            st.stop()

        # =================================================
        # PROCESS VIDEO FRAME-BY-FRAME
        # =================================================

        progress_bar = st.progress(0)

        status_text = st.empty()

        processed_frames = 0

        try:

            results_stream = model.track(
                source=temp_input.name,
                conf=0.25,
                tracker="bytetrack.yaml",
                persist=True,
                stream=True,
                verbose=False
            )

            for result in results_stream:

                # Draw bounding boxes and tracking IDs
                annotated_frame = result.plot()

                # Write frame to output video
                writer.write(
                    annotated_frame
                )

                processed_frames += 1

                # Update progress
                if total_frames > 0:

                    progress = min(
                        processed_frames / total_frames,
                        1.0
                    )

                    progress_bar.progress(
                        progress
                    )

                    status_text.text(
                        f"Processing frame "
                        f"{processed_frames}/{total_frames}"
                    )

        except Exception as e:

            writer.release()

            st.error(
                "Video tracking failed."
            )

            st.exception(e)

            try:
                os.unlink(temp_input.name)
            except Exception:
                pass

            st.stop()

        finally:

            writer.release()

        progress_bar.progress(1.0)

        status_text.text(
            f"Processing completed: "
            f"{processed_frames} frames"
        )

        st.success(
            "YOLO detection and tracking completed!"
        )

        # =================================================
        # CHECK OUTPUT VIDEO
        # =================================================

        if not os.path.exists(output_video):

            st.error(
                "Output video was not created."
            )

            try:
                os.unlink(temp_input.name)
            except Exception:
                pass

            st.stop()

        output_size = os.path.getsize(
            output_video
        )

        if output_size == 0:

            st.error(
                "Output video is empty."
            )

            try:
                os.unlink(temp_input.name)
            except Exception:
                pass

            st.stop()

        st.info(
            f"Processed {processed_frames} frames successfully."
        )

        # =================================================
        # CONVERT TO H.264
        # =================================================

        browser_video = os.path.join(
            tempfile.gettempdir(),
            "yolo_processed_h264.mp4"
        )

        if os.path.exists(browser_video):

            try:
                os.remove(browser_video)
            except Exception:
                pass

        try:

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg_path,
                "-y",
                "-i",
                output_video,
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
                "Preparing video for browser playback..."
            ):

                process = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            # =================================================
            # CHECK FFMPEG
            # =================================================

            if process.returncode != 0:

                st.error(
                    "Video conversion failed."
                )

                st.code(
                    process.stderr[-4000:]
                )

            elif not os.path.exists(browser_video):

                st.error(
                    "Converted video file was not created."
                )

            else:

                # =================================================
                # DISPLAY VIDEO
                # =================================================

                st.subheader(
                    "Processed Video"
                )

                # IMPORTANT:
                # Pass the file path directly.
                # Do not load the entire video into RAM.

                st.video(
                    browser_video,
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