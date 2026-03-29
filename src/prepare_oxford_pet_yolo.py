import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SEED = 42

SELECTED_BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
]

CLASS_TO_ID = {breed: i for i, breed in enumerate(SELECTED_BREEDS)}

IMAGES_DIR = Path("data/images")
XML_DIR = Path("data/annotations/xmls")
OUT_DIR = Path("data/oxford_pet_yolo")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    obj = root.find("object")
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    return filename, width, height, xmin, ymin, xmax, ymax


def get_breed_from_filename(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    breed = "_".join(parts[:-1])
    return breed


def to_yolo_format(width, height, xmin, ymin, xmax, ymax):
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_old_files():
    for split in ["train", "val", "test"]:
        for folder in [OUT_DIR / "images" / split, OUT_DIR / "labels" / split]:
            for file_path in folder.glob("*"):
                if file_path.is_file():
                    file_path.unlink()


def main():
    random.seed(SEED)
    ensure_dirs()
    clear_old_files()

    breed_to_examples = {breed: [] for breed in SELECTED_BREEDS}

    for xml_file in sorted(XML_DIR.glob("*.xml")):
        try:
            filename, width, height, xmin, ymin, xmax, ymax = parse_xml(xml_file)
        except Exception as e:
            print(f"Skipping {xml_file.name}: {e}")
            continue

        breed = get_breed_from_filename(filename)

        if breed not in CLASS_TO_ID:
            continue

        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            print(f"Missing image for {filename}, skipping.")
            continue

        breed_to_examples[breed].append(
            {
                "filename": filename,
                "width": width,
                "height": height,
                "class_id": CLASS_TO_ID[breed],
                "bbox": (xmin, ymin, xmax, ymax),
            }
        )

    for breed, examples in breed_to_examples.items():
        random.shuffle(examples)

        n = len(examples)
        train_end = int(TRAIN_RATIO * n)
        val_end = train_end + int(VAL_RATIO * n)

        splits = {
            "train": examples[:train_end],
            "val": examples[train_end:val_end],
            "test": examples[val_end:],
        }

        print(
            f"{breed}: total={n}, "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )

        for split, items in splits.items():
            for item in items:
                filename = item["filename"]
                class_id = item["class_id"]
                width = item["width"]
                height = item["height"]
                xmin, ymin, xmax, ymax = item["bbox"]

                x_center, y_center, box_width, box_height = to_yolo_format(
                    width, height, xmin, ymin, xmax, ymax
                )

                src_img = IMAGES_DIR / filename
                dst_img = OUT_DIR / "images" / split / filename
                shutil.copy2(src_img, dst_img)

                label_name = Path(filename).stem + ".txt"
                dst_label = OUT_DIR / "labels" / split / label_name

                with open(dst_label, "w") as f:
                    f.write(
                        f"{class_id} "
                        f"{x_center:.6f} {y_center:.6f} "
                        f"{box_width:.6f} {box_height:.6f}\n"
                    )

    yaml_path = OUT_DIR / "pets.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write(f"nc: {len(SELECTED_BREEDS)}\n")
        f.write("names:\n")
        for i, breed in enumerate(SELECTED_BREEDS):
            f.write(f"  {i}: {breed}\n")

    print(f"\nDone. YOLO dataset written to: {OUT_DIR}")
    print(f"YAML file written to: {yaml_path}")


if __name__ == "__main__":
    main()
     
               
             
