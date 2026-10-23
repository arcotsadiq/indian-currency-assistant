
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data/currency_split/currency.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project="runs/currency",
        name="generalization_v2",
        exist_ok=False,
        plots=False,
    )

    print("\nTraining completed.")
    print("Check runs/currency/generalization_v2/weights/best.pt")


if __name__ == "__main__":
    main()
