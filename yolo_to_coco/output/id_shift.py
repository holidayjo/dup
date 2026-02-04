import json
from pathlib import Path

def remap_coco_categories(
    input_json,
    output_json,
    class_names
):
    with open(input_json, "r") as f:
        coco = json.load(f)

    # Build mapping: old_id -> new_id
    old_categories = coco["categories"]
    name_to_new_id = {name: i for i, name in enumerate(class_names)}

    old_id_to_new_id = {}
    new_categories = []

    for cat in old_categories:
        name = cat["name"]
        if name not in name_to_new_id:
            raise ValueError(f"Unknown category name: {name}")
        new_id = name_to_new_id[name]
        old_id_to_new_id[cat["id"]] = new_id
        new_categories.append({
            "id": new_id,
            "name": name
        })

    # Update annotations
    for ann in coco["annotations"]:
        old_id = ann["category_id"]
        if old_id not in old_id_to_new_id:
            raise ValueError(f"Invalid category_id {old_id}")
        ann["category_id"] = old_id_to_new_id[old_id]

    coco["categories"] = new_categories

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved fixed COCO file to: {output_json}")


if __name__ == "__main__":
    classes = ["U", "D", "P"]

    remap_coco_categories(
        input_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/train.json",
        output_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/train_fixed.json",
        class_names=classes
    )

    remap_coco_categories(
        input_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/test.json",
        output_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/test_fixed.json",
        class_names=classes
    )

    remap_coco_categories(
        input_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/val.json",
        output_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/val_fixed.json",
        class_names=classes
    )

    remap_coco_categories(
        input_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/TrainVal.json",
        output_json="/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/TrainVal_fixed.json",
        class_names=classes
    )
