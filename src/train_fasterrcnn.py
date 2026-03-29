import time
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import PennFudanDataset, get_train_transforms, get_eval_transforms


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}: train loss = {avg_loss:.4f}")


def evaluate_loss(model, dataloader, device):
    model.train()  # needed because detection models return losses only in train mode
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    full_train_dataset = PennFudanDataset(
        root="data/PennFudanPed",
        transforms=get_train_transforms(image_size=512)
    )

    full_eval_dataset = PennFudanDataset(
        root="data/PennFudanPed",
        transforms=get_eval_transforms(image_size=512)
    )

    n = len(full_train_dataset)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(n), [train_size, val_size, test_size], generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_eval_dataset, val_indices.indices)
    test_dataset = torch.utils.data.Subset(full_eval_dataset, test_indices.indices)

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    model = get_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 12
    best_val_loss = float("inf")

    start_time = time.time()

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch + 1}: val loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/fasterrcnn_pennfudan_best.pth")
            print("Saved best model.")

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    print("Training complete.")
    print("Test set size:", len(test_dataset))


if __name__ == "__main__":
    main()
