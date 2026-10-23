
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pyttsx3


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "currency"
    / "test"
    / "weights"
    / "best.pt"
)

CONFIDENCE_THRESHOLD = 0.05
RELIABLE_PREDICTION_THRESHOLD = 0.50


# --------------------------------------------------
# 2. Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Indian Currency Assistant",
    page_icon="💵",
    layout="centered",
)

st.title("💵 Indian Currency Detection Assistant")

st.write(
    "An AI-based prototype that attempts to identify "
    "Indian currency denominations and announce its "
    "prediction through speech."
)

st.info(
    "This is a student project, not a validated assistive "
    "device. Never rely on its prediction alone to identify "
    "money. Always verify the denomination independently."
)


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


if not MODEL_PATH.is_file():
    st.error(f"Trained model not found:\n{MODEL_PATH}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# --------------------------------------------------
# 4. Upload currency image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a currency note image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded currency image",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Unable to open this image: {exc}")
        st.stop()

    if st.button("Detect Currency", type="primary"):
        # Clear the previous prediction before a new test.
        st.session_state.pop("currency_result", None)
        st.session_state.pop("currency_error", None)

        with st.spinner("Analyzing the image..."):
            try:
                results = model.predict(
                    source=image,
                    imgsz=640,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=0.45,
                    device="cpu",
                    verbose=False,
                )

                result = results[0]
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    st.session_state["currency_error"] = (
                        "No currency note was detected. Try a clear, "
                        "well-lit photo with the entire note visible."
                    )
                else:
                    # Sort detections from highest to lowest confidence.
                    confidences = boxes.conf.tolist()
                    order = sorted(
                        range(len(confidences)),
                        key=lambda i: confidences[i],
                        reverse=True,
                    )

                    candidates = []

                    # Keep the strongest prediction for each denomination.
                    # This removes repeated labels from the candidate list;
                    # it does not prove that the chosen denomination is correct.
                    seen_classes = set()

                    for index in order:
                        class_id = int(boxes.cls[index].item())
                        confidence = float(
                            boxes.conf[index].item()
                        )
                        denomination = result.names[class_id]

                        if class_id not in seen_classes:
                            candidates.append(
                                {
                                    "denomination": denomination,
                                    "confidence": confidence,
                                }
                            )
                            seen_classes.add(class_id)

                    best = candidates[0]

                    st.session_state["currency_result"] = {
                        "denomination": best["denomination"],
                        "confidence": best["confidence"],
                        "candidates": candidates,
                        "annotated_image": (
                            result.plot()[:, :, ::-1]
                        ),
                    }

            except Exception as exc:
                st.session_state["currency_error"] = (
                    f"Detection failed: {exc}"
                )


# --------------------------------------------------
# 5. Display prediction
# --------------------------------------------------

if st.session_state.get("currency_error"):
    st.error(st.session_state["currency_error"])

detection = st.session_state.get("currency_result")

if detection is not None:
    denomination = detection["denomination"]
    confidence = detection["confidence"]

    st.subheader("Detection result")

    st.image(
        detection["annotated_image"],
        caption="Model prediction with bounding boxes",
        use_container_width=True,
    )

    st.metric(
        label="Top model prediction",
        value=denomination,
        delta=f"{confidence:.1%} confidence",
        delta_color="off",
    )

    # Confidence indicates model certainty, not actual accuracy.
    if confidence < RELIABLE_PREDICTION_THRESHOLD:
        st.error(
            "UNCERTAIN RESULT: The model's confidence is too low "
            "to trust this prediction. Do not use this result to "
            "identify the note. Try another image and verify "
            "the denomination independently."
        )
    else:
        st.warning(
            "The model has a stronger prediction, but it can still "
            "be wrong. Independently verify the denomination."
        )

    # Show other denomination candidates when available.
    with st.expander("View alternative model predictions"):
        for candidate in detection["candidates"]:
            st.write(
                f"{candidate['denomination']}: "
                f"{candidate['confidence']:.1%}"
            )

    # --------------------------------------------------
    # 6. Voice announcement
    # --------------------------------------------------

    if st.button("🔊 Speak Result"):
        try:
            engine = pyttsx3.init()

            if confidence < RELIABLE_PREDICTION_THRESHOLD:
                message = (
                    "Warning. The prediction is uncertain. "
                    f"The model suggests {denomination}, "
                    f"with {confidence:.0%} confidence. "
                    "Do not rely on this result. "
                    "Please verify the denomination independently."
                )
            else:
                message = (
                    f"The model predicts {denomination}, "
                    f"with {confidence:.0%} confidence. "
                    "This result may be incorrect. "
                    "Please verify the denomination independently."
                )

            engine.say(message)
            engine.runAndWait()
            engine.stop()

            st.success("Voice announcement completed.")

        except Exception as exc:
            st.error(f"Speech playback failed: {exc}")


# --------------------------------------------------
# 7. Supported denominations
# --------------------------------------------------

with st.expander("Supported denominations"):
    st.write("₹10, ₹20, ₹50, ₹100, ₹200 and ₹500.")
    st.caption(
        "The trained model does not include ₹2000. "
        "Performance on different note designs and real-world "
        "photos has not yet been validated."
    )


# --------------------------------------------------
# 8. Project information
# --------------------------------------------------

st.divider()

st.caption(
    "Indian Currency Detection for Visually Impaired People | "
    "YOLO object detection | Streamlit | Python"
)
