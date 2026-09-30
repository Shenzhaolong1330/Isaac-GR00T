"""Import/version check; optional tiny CUDA calculation, no model or robot loading."""
import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--report", default="deployment_records/environment.json")
    a = p.parse_args()
    report = {"python": platform.python_version(), "packages": {}}
    for name, module in [("torch", "torch"), ("transformers", "transformers"),
                         ("torchcodec", "torchcodec"), ("flash-attn", "flash_attn"),
                         ("deepspeed", "deepspeed"), ("gr00t", "gr00t")]:
        mod = importlib.import_module(module)
        report["packages"][name] = {"version": importlib.metadata.version(name), "path": mod.__file__}
    import torch
    if a.cuda:
        x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        assert torch.isfinite(y).all().item()
        report["cuda"] = {"device": torch.cuda.get_device_name(), "version": torch.version.cuda}
    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
