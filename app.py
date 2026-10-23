
import json
import os
from pathlib import Path

import gdown
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Indian Currency Assistant",
    page_icon="💵",
)

st.title("Indian Currency Assistant")
st.write("Indian banknote detection with English voice assistance.")

MODEL_PATH = Path("best.pt")
MODEL_URL = (
    "https://drive.google.com/uc?id="
    "1-TbbTaMc7EWI-H36IioKaAVaEtGXER9E"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        result = gdown.download(
    MODEL_URL,
    str(MODEL_PATH),
    quiet=False,
)
        if not result or not MODEL_PATH.exists():
            raise RuntimeError(
                "Could not download the model from Google Drive. "
                "Check the sharing permissions and try again."
            )

    return YOLO(str(MODEL_PATH))


try:
    with st.spinner("Loading currency recognition model..."):
        model = load_model()
    st.success("Currency recognition model loaded.")
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.stop()

image_file = st.camera_input("Capture an Indian banknote")

if image_file is None:
    image_file = st.file_uploader(
        "Or upload a banknote image",
        type=["jpg", "jpeg", "png"],
    )

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Input image", width=400)

    if st.button("Identify Currency", type="primary"):
        with st.spinner("Analyzing image..."):
            results = model.predict(image, conf=0.25, verbose=False)
            result = results[0]
            names = result.names
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                message = (
                    "No currency note was detected. "
                    "Try a clearer image with good lighting."
                )
                st.warning(message)
            else:
                detected = []
                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    label = str(names[class_id])
                    detected.append((label, confidence))

                st.image(
                    result.plot(),
                    caption="Detection results",
                    width=600,
                )

                st.subheader("Detected currency")
                for label, confidence in detected:
                    st.write(
                        f"**{label}** — confidence {confidence:.1%}"
                    )

                labels_text = ", ".join(
                    label for label, _ in detected
                )
                message = f"Detected currency: {labels_text}."

            st.session_state["speech_text"] = message

if st.button("Read Result Aloud"):
    speech_text = st.session_state.get("speech_text", "")
    if speech_text:
        safe_text = json.dumps(speech_text)
        st.components.v1.html(
            f"""
            <script>
            const message = {safe_text};
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.lang = "en-IN";
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
            </script>
            """,
            height=0,
        )
    else:
        st.warning("Identify an image first.")
