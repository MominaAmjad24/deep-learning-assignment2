from dataset import PennFudanDataset, get_train_transforms

dataset = PennFudanDataset(
    root="data/PennFudanPed",
    transforms=get_train_transforms(image_size=512)
)

print("Dataset size:", len(dataset))

image, target = dataset[0]
print("Image shape:", image.shape)
print("Target keys:", target.keys())
print("Boxes shape:", target["boxes"].shape)
print("Labels:", target["labels"])

