"""Download gated Cosmos ONLY from the official Hugging Face endpoint after authorization."""
import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import HfApi, snapshot_download

if __name__ == "__main__":
    repo = "nvidia/Cosmos-Reason2-2B"
    api = HfApi(endpoint="https://huggingface.co")
    info = api.model_info(repo)
    destination = Path("checkpoints/Cosmos-Reason2-2B")
    snapshot_download(repo, revision=info.sha, local_dir=destination,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "README.md", "LICENSE*"],
        max_workers=2)
    (destination / "download_manifest.json").write_text(json.dumps({
        "repo_id": repo, "revision": info.sha, "endpoint": "https://huggingface.co"
    }, indent=2))
    print(destination)
