import os
import time
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import PennFudanDataset, get_eval_transforms


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def split_indices(n, train_ratio=0.7, val_ratio=0.15, seed=42):
    torch.manual_seed(seed)
    indices = torch.randperm(n).tolist()

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices


def draw_boxes(image, boxes, scores, save_path, threshold=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    kept = 0
    for box, score in zip(boxes, scores):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{score:.2f}", fontsize=8, color='red')
            kept += 1

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return kept


def main():
    os.makedirs("outputs/predictions", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    dataset = PennFudanDataset(
        root="data/PennFudanPed",
        transforms=get_eval_transforms(image_size=512)
    )

    n = len(dataset)
    _, _, test_indices = split_indices(n)
    test_dataset = Subset(dataset, test_indices)

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("outputs/fasterrcnn_pennfudan_best.pth", map_location=device))
    model.to(device)
    model.eval()

    total_time = 0.0
    total_images = len(test_dataset)

    for i in range(min(5, total_images)):
        image, target = test_dataset[i]
        image_input = image.to(device)

        start = time.time()
        with torch.no_grad():
            prediction = model([image_input])[0]
        end = time.time()

        inference_time = end - start
        total_time += inference_time

        boxes = prediction["boxes"].cpu()
        scores = prediction["scores"].cpu()

        kept = draw_boxes(
            image.cpu(),
            boxes,
            scores,
            save_path=f"outputs/predictions/prediction_{i}.png",
            threshold=0.5
        )

        print(f"Image {i}: inference time = {inference_time:.4f}s, kept boxes = {kept}")

    avg_time = total_time / min(5, total_images)
    ips = 1.0 / avg_time if avg_time > 0 else 0.0

    print(f"\nAverage inference time per image: {avg_time:.4f} seconds")
    print(f"Inference speed: {ips:.2f} images/second")


if __name__ == "__main__":
    main()
