from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/currency/test/weights/best.pt"
)

image_path = (
    r"C:\Users\arcot\Downloads\Indian-Currency-DataSet-main"
    r"\Indian-Currency-DataSet-main\images\val\0.jpg"
)

for size in [320, 640]:
    print(f"\n--- Testing image size {size} ---")

    results = model.predict(
        source=image_path,
        imgsz=size,
        conf=0.25,
        device="cpu",
        save=False,
        verbose=False,
    )

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("No detections.")
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            print(
                f"{result.names[class_id]}: {confidence:.1%}"
            )

