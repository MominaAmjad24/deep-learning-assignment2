from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/oxford_pet_yolo/pets.yaml",
        epochs=20,
        imgsz=512,
        batch=8,
        project="outputs",
        name="yolov8_pets",
        pretrained=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
