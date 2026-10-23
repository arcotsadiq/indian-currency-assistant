from pathlib import Path
import pyttsx3
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/runs/currency/test/weights/best.pt")


def speak(message):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(message)
    engine.runAndWait()
    engine.stop()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    image_input = input(
        "Enter the full path of a currency image: "
    ).strip().strip('"')

    image_path = Path(image_input).expanduser()

    if not image_path.is_file():
        print("Image file not found. Please check the path.")
        return

    if image_path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        print("Please select a JPG, JPEG, PNG, BMP, or WEBP image.")
        return

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(image_path),
        imgsz=320,
        conf=0.50,
        device="cpu",
        save=True,
        project="runs/currency",
        name="prediction",
        exist_ok=True,
    )

    detections = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                currency = result.names[class_id]
                detections.append((currency, confidence))

    if not detections:
        message = (
            "No currency note detected confidently. "
            "Please try another photo with better lighting."
        )
        print(message)
        speak(message)
        return

    # Announce the highest-confidence detected note.
    currency, confidence = max(detections, key=lambda item: item[1])
    amount = currency.replace(" INR", "")

    print(f"Detected currency: {currency}")
    print(f"Model confidence: {confidence:.1%}")

    message = f"Indian rupees {amount} detected."
    print(f"Speaking: {message}")
    speak(message)

    print("Annotated image saved under runs/detect/runs/currency/prediction")


if __name__ == "__main__":
    main()
