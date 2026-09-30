"""Expose an authorized local Cosmos snapshot to HF offline loading without copying weights."""
import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="checkpoints")
    a = p.parse_args()
    root = Path(a.root).resolve()
    source = root / "Cosmos-Reason2-2B"
    manifest = json.loads((source / "download_manifest.json").read_text())
    revision = manifest["revision"]
    dest = root / "hub/models--nvidia--Cosmos-Reason2-2B"
    snapshot = dest / "snapshots" / revision
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.symlink_to(source, target_is_directory=True)
    elif snapshot.resolve() != source:
        raise FileExistsError(snapshot)
    (dest / "refs").mkdir(exist_ok=True)
    (dest / "refs/main").write_text(revision)
    print(f"export HF_HUB_CACHE={root / 'hub'}")
