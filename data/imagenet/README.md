1. Download Imagenet-1k from HF using `imagenet-hf.py`
2. Unzip to validation dataset to `/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/imagenet1k-val`
3. Separate by synset folders using `organize.py`
4. Run `imagenet-subset.sh`
5. Convert to hf datasets with `convert-dataset-hf.py`
