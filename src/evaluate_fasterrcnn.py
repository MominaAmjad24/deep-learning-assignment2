import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

from dataset import PennFudanDataset, get_eval_transforms


def collate_fn(batch):
    return tuple(zip(*batch))


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


def compute_image_stats(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5, score_threshold=0.5):
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = box_iou(pred_boxes, gt_boxes)

    matched_gt = set()
    tp = 0
    fp = 0

    for i in range(len(pred_boxes)):
        max_iou, max_j = torch.max(ious[i], dim=0)
        if max_iou.item() >= iou_threshold and max_j.item() not in matched_gt:
            tp += 1
            matched_gt.add(max_j.item())
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_ap(all_predictions, all_targets, iou_threshold=0.5):
    scores = []
    matches = []
    total_gt = 0

    for preds, targets in zip(all_predictions, all_targets):
        pred_boxes = preds["boxes"].cpu()
        pred_scores = preds["scores"].cpu()
        gt_boxes = targets["boxes"].cpu()

        total_gt += len(gt_boxes)

        used_gt = set()
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)

            for i in range(len(pred_boxes)):
                score = pred_scores[i].item()
                max_iou, max_j = torch.max(ious[i], dim=0)

                if max_iou.item() >= iou_threshold and max_j.item() not in used_gt:
                    scores.append(score)
                    matches.append(1)
                    used_gt.add(max_j.item())
                else:
                    scores.append(score)
                    matches.append(0)
        else:
            for i in range(len(pred_boxes)):
                scores.append(pred_scores[i].item())
                matches.append(0)

    if len(scores) == 0:
        return 0.0

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    matches = [matches[i] for i in sorted_indices]

    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []

    for match in matches:
        if match == 1:
            tp_cum += 1
        else:
            fp_cum += 1

        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / total_gt if total_gt > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r

    return ap


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    dataset = PennFudanDataset(
        root="data/PennFudanPed",
        transforms=get_eval_transforms(image_size=512)
    )

    n = len(dataset)
    _, _, test_indices = split_indices(n)
    test_dataset = Subset(dataset, test_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("outputs/fasterrcnn_pennfudan_best.pth", map_location=device))
    model.to(device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            output = outputs[0]
            target = targets[0]

            pred_boxes = output["boxes"].cpu()
            pred_scores = output["scores"].cpu()
            gt_boxes = target["boxes"].cpu()

            tp, fp, fn = compute_image_stats(pred_boxes, pred_scores, gt_boxes)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            all_predictions.append(output)
            all_targets.append(target)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    map50 = compute_ap(all_predictions, all_targets, iou_threshold=0.5)

    print("\nEvaluation Results on Penn-Fudan Test Set")
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"mAP@0.5:         {map50:.4f}")


if __name__ == "__main__":
    main()
