from huggingface_hub import snapshot_download
from dotenv import dotenv_values

REPO_ID = "ILSVRC/imagenet-1k"
CACHE_DIR = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/"
TOKEN= dotenv_values("/workspaces/ASE-Model-Retrieval/data/imagenet/.env")["HF_TOKEN"]

snapshot_download(repo_id=REPO_ID, cache_dir=CACHE_DIR, repo_type="dataset", token=TOKEN)