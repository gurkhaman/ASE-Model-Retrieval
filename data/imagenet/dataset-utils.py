import json
import os
import random
import shutil

IMAGE_DIR = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/12_classes"
SUPERCLASS_OUTPUT_DIR = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/superclass_datasets"
)
SUBCLASS_OUTPUT_DIR = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/subclass_datasets"
)
SUPERCLASS_SUBCLASS_MAPPING_PATH = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/imagenet_mapping.json"
)
NUM_IMAGES_SUPERCLASS = 250
NUM_DATASETS = 300
NUM_SUBCLASSES_PER_DATASET = 5
NUM_IMAGES_PER_SUBCLASS = 50
SEED = 42


def load_imagenet_mapping():
    with open(SUPERCLASS_SUBCLASS_MAPPING_PATH, "r") as f:
        return json.load(f)


def create_subclass_datasets():
    imagenet_mapping = load_imagenet_mapping()

    all_subclasses = []
    subclass_to_path = {}

    for superclass, subclasses in imagenet_mapping.items():
        for subclass_name in subclasses.values():
            subclass_path = os.path.join(IMAGE_DIR, superclass, subclass_name)
            if os.path.exists(subclass_path):
                all_subclasses.append(subclass_name)
                subclass_to_path[subclass_name] = subclass_path

    if len(all_subclasses) < NUM_SUBCLASSES_PER_DATASET:
        raise ValueError(
            "Not enough subclasses to form a dataset with 5 subclasses each."
        )
    random.seed(SEED)
    unique_datasets = set()
    while len(unique_datasets) < NUM_DATASETS:
        random_combination = tuple(
            sorted(random.sample(all_subclasses, NUM_SUBCLASSES_PER_DATASET))
        )
        unique_datasets.add(random_combination)

    os.makedirs(SUBCLASS_OUTPUT_DIR, exist_ok=True)

    for i, dataset in enumerate(unique_datasets):
        dataset_dir = os.path.join(SUBCLASS_OUTPUT_DIR, f"dataset_{i + 1}")
        os.makedirs(dataset_dir, exist_ok=True)

        for subclass in dataset:
            subclass_path = subclass_to_path[subclass]
            images = [
                os.path.join(subclass_path, img)
                for img in os.listdir(subclass_path)
                if img.endswith(".JPEG")
            ]

            if len(images) < NUM_IMAGES_PER_SUBCLASS:
                raise ValueError(f"Subclass {subclass} does not have enough images.")

            selected_images = random.sample(images, NUM_IMAGES_PER_SUBCLASS)

            for img_path in selected_images:
                shutil.copy(
                    img_path, os.path.join(dataset_dir, os.path.basename(img_path))
                )

        print(f"Dataset {i + 1} created with subclasses: {dataset}")

    print("All 300 random datasets created successfully!")


def create_superclass_datasets():
    imagenet_mapping = load_imagenet_mapping()
    os.makedirs(SUPERCLASS_OUTPUT_DIR, exist_ok=True)

    for superclass, subclasses in imagenet_mapping.items():
        superclass_images = []

        for subclass_name in subclasses.values():
            subclass_path = os.path.join(IMAGE_DIR, superclass, subclass_name)
            if os.path.exists(subclass_path):
                images = [
                    os.path.join(subclass_path, img)
                    for img in os.listdir(subclass_path)
                    if img.endswith(".JPEG")
                ]
                superclass_images.extend(images)

        random.seed(SEED)
        selected_images = random.sample(
            superclass_images, min(len(superclass_images), NUM_IMAGES_SUPERCLASS)
        )

        output_superclass_dir = os.path.join(SUPERCLASS_OUTPUT_DIR, superclass)
        os.makedirs(output_superclass_dir, exist_ok=True)

        for img_path in selected_images:
            shutil.copy(
                img_path,
                os.path.join(output_superclass_dir, os.path.basename(img_path)),
            )

        print(
            f"Superclass '{superclass}' processed: {len(selected_images)} images saved."
        )

    print("All superclass datasets created successfully!")


if __name__ == "__main__":
    create_superclass_datasets()
    create_subclass_datasets()