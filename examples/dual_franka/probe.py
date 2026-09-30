"""Read-only probe of the native ZMQ policy server; never opens robot hardware."""
import argparse
from gr00t.policy.server_client import PolicyClient
from examples.dual_franka.run import read_config

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="examples/dual_franka/server.yaml")
    a = p.parse_args()
    cfg = read_config(a.config)
    with PolicyClient(host=cfg.get("host", "127.0.0.1"), port=cfg.get("port", 5555)) as client:
        if not client.ping():
            raise RuntimeError("Policy server did not respond")
        print(client.get_modality_config())
