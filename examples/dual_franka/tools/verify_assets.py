"""Verify copied checkpoint files against the source SHA256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="deployment_records/source_hashes.json")
    parser.add_argument("--root", default="checkpoints")
    parser.add_argument("--manifest", help="Inference export manifest; --root is its directory")
    parser.add_argument("--report", default="deployment_records/transfer_verified.json")
    args = parser.parse_args()
    reference = (
        {".": json.loads(Path(args.manifest).read_text())["files"]}
        if args.manifest
        else json.loads(Path(args.reference).read_text())
    )
    results = {}
    for model, files in reference.items():
        results[model] = {}
        for name, expected in files.items():
            path = Path(args.root) / model / name
            if path.stat().st_size != expected["size"]:
                raise ValueError(f"Size mismatch: {path}")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected["sha256"]:
                raise ValueError(f"SHA256 mismatch: {path}")
            results[model][name] = expected
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"verified": True, "files": results}, indent=2))
    print(report)
