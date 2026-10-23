
import streamlit as st
import json

st.set_page_config(
    page_title="Indian Currency Assistant",
    page_icon="💵",
    layout="centered"
)

st.title("Indian Currency Assistant")
st.subheader("Accessible currency recognition")
st.write(
    "Upload an Indian banknote image or use your camera. "
    "Voice output is available in English."
)

image = st.camera_input("Capture a banknote")

if image is None:
    image = st.file_uploader(
        "Or upload a banknote image",
        type=["jpg", "jpeg", "png"]
    )

if st.button("Identify Currency", type="primary"):
    if image is None:
        st.warning("Please upload or capture an image.")
    else:
        st.image(image, caption="Captured banknote", width=300)
        st.info(
            "Image received. The currency recognition model "
            "is not connected yet, so the denomination cannot "
            "be identified at this stage."
        )

        st.session_state["speech_text"] = (
            "Image received. The currency recognition model "
            "is not connected yet."
        )

if st.button("Read Result Aloud"):
    text = st.session_state.get("speech_text", "")
    if text:
        st.components.v1.html(
            f"""
            <script>
            const text = {json.dumps(text)};
            window.speechSynthesis.cancel();
            const speech = new SpeechSynthesisUtterance(text);
            speech.lang = "en-IN";
            window.speechSynthesis.speak(speech);
            </script>
            """,
            height=0
        )
    else:
        st.warning("Please identify an image first.")
