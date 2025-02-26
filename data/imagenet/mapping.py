import json
import re
from collections import defaultdict

SCRIPT_PATH = "/workspaces/ASE-Model-Retrieval/data/imagenet/imagenet-subset.sh"
JSON_PATH = "/workspaces/ASE-Model-Retrieval/data/imagenet/imagenet_mapping.json"

pattern = re.compile(r"ln -s \$DATASET_PATH/(n\d+) \$OUTPUT_PATH/([^/]+)/([^/\n]+)")
mapping = defaultdict(dict)

with open(SCRIPT_PATH, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            class_id, superclass, name = match.groups()
            mapping[superclass][class_id] = name

mapping = dict(mapping)

with open(JSON_PATH, "w") as json_file:
    json.dump(mapping, json_file, indent=4)
