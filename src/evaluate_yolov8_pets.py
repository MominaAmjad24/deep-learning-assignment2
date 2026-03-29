from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/outputs/yolov8_pets/weights/best.pt")

    metrics = model.val(
        data="data/oxford_pet_yolo/pets.yaml",
        split="test",
        imgsz=512
    )

    print("\nEvaluation complete.")
    print(metrics)


if __name__ == "__main__":
    main()
