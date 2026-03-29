from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/outputs/yolov8_pets/weights/best.pt")

    model.predict(
        source="data/oxford_pet_yolo/images/test",
        imgsz=512,
        conf=0.25,
        save=True,
        project="outputs",
        name="yolov8_pets_predictions"
    )


if __name__ == "__main__":
    main()

