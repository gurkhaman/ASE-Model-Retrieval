from pathlib import Path
import shutil
import re


# I thought file names didn't have synset labels... Keeping this here just in case.
def organize_by_synset_external(val_img_dir, synset_labels, output_dir):
    with open(synset_labels, "r") as f:
        synsets = [line.strip() for line in f]

    created_dirs = set()

    for idx, synset in enumerate(synsets, start=1):
        image_name = f"ILSVRC2012_val_{idx:08d}.JPEG"
        src_path = Path(val_img_dir) / image_name

        dest_dir = Path(output_dir) / synset

        if dest_dir not in created_dirs:
            dest_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(dest_dir)

        dest_path = dest_dir / image_name
        shutil.move(src_path, dest_path)

    print("Images successfully organized")


def organize_by_synset_filename(val_img_dir, output_dir):
    synset_pattern = re.compile(r"n[0-9]{8}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in Path(val_img_dir).glob("*.JPEG"):
        match = synset_pattern.search(image_path.name)
        if match:
            synset = match.group()
            synset_dir = output_dir / synset
            synset_dir.mkdir(parents=True, exist_ok=True)

            dest_path = synset_dir / image_path.name
            shutil.move(image_path, dest_path)
        else:
            print("Couldn't match file: ", image_path)


VAL_IMAGES_DIR = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/imagenet1k-val"
SYNSET_LABELS_FILE = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/val_synset_labels.txt"
)
OUTPUT_DIR = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/imagenet-synset"

organize_by_synset_filename(val_img_dir=VAL_IMAGES_DIR, output_dir=OUTPUT_DIR)
