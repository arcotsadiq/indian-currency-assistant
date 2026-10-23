
import gradio as gr
import json


def detect_currency(image):
    if image is None:
        return "Please upload an image or capture a currency note."

    # Model integration will be added in the next stage.
    return (
        "Image received successfully. The currency recognition model "
        "is not connected yet, so a denomination cannot be predicted."
    )


def prepare_speech(text):
    return json.dumps(text)


with gr.Blocks(title="Indian Currency Assistant") as demo:
    gr.Markdown(
        """
        # Indian Currency Assistant
        ### Accessible currency recognition for visually impaired people

        Upload a picture or use your camera to capture an Indian banknote.
        """
    )

    with gr.Row():
        image = gr.Image(
            label="Upload or capture a banknote",
            sources=["upload", "webcam"],
            type="pil"
        )

    identify_button = gr.Button(
        "Identify Currency",
        variant="primary"
    )

    result = gr.Textbox(
        label="Result",
        interactive=False,
        lines=3
    )

    identify_button.click(
        fn=detect_currency,
        inputs=image,
        outputs=result
    )

    speak_button = gr.Button("Read Result Aloud")

    speak_button.click(
        fn=prepare_speech,
        inputs=result,
        outputs=None,
        js="""
        (text) => {
            if (!text || !text.trim()) return;
            window.speechSynthesis.cancel();
            const speech = new SpeechSynthesisUtterance(text);
            speech.lang = "en-IN";
            window.speechSynthesis.speak(speech);
        }
        """
    )

demo.launch()
