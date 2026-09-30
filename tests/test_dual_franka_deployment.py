import json
from pathlib import Path
import tempfile
import unittest

import importlib.util

def load_example(name):
    path = Path(__file__).parents[1] / "examples/dual_franka" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

read_config = load_example("run").read_config
prepare = load_example("prepare_physbrain").prepare


class DeploymentConfigTest(unittest.TestCase):
    def test_yaml_inheritance_and_initialization(self):
        cfg = read_config("examples/dual_franka/train_physbrain.yaml")
        self.assertIsNone(cfg["training"]["start_from_checkpoint"])
        self.assertEqual(cfg["training"]["global_batch_size"], 64)
        self.assertEqual(cfg["model"]["model_name"], "checkpoints/PhysBrain1.5-2B-gr00t")
        self.assertFalse(cfg["model"]["use_relative_action"])

    def test_job_resume_keeps_schedule_and_run(self):
        first = read_config("examples/dual_franka/train_job.yaml")
        resumed = read_config("examples/dual_franka/train_job_resume.yaml")
        self.assertEqual(first["training"]["num_gpus"], 4)
        self.assertEqual(first["training"]["global_batch_size"] * first["training"]["gradient_accumulation_steps"], 64)
        self.assertEqual(first["training"]["max_steps"], 1000)
        self.assertEqual(first["deployment"]["stop_after_step"], 500)
        self.assertIsNone(resumed["deployment"]["stop_after_step"])
        self.assertTrue(resumed["training"]["resume_from_checkpoint"])
        self.assertEqual(first["training"]["output_dir"], resumed["training"]["output_dir"])
        self.assertEqual(first["deployment"]["wandb_id"], resumed["deployment"]["wandb_id"])
        self.assertFalse(first["data"]["allow_padding"])

    def test_rank_zero_metrics_do_not_change_trainer_control(self):
        from types import SimpleNamespace
        # Load the actual pure logging method without initializing model imports.
        import ast
        source = Path(__file__).parents[1] / "examples/dual_franka/training_support.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AcceptanceCallback")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "log_metrics")
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        callback = SimpleNamespace()
        control = SimpleNamespace(should_log=True)
        def forbidden_log(*args, **kwargs):
            control.should_log = False
            self.fail("Rank-zero-only metrics must not invoke Trainer.log")
        callback.trainer = SimpleNamespace(
            state=SimpleNamespace(global_step=2, log_history=[]),
            control=control, log=forbidden_log)
        callback.config = SimpleNamespace(training=SimpleNamespace(use_wandb=False))
        scope["log_metrics"](callback, {"update/vision/sample_max_change": 0.1})
        self.assertTrue(control.should_log)
        self.assertEqual(callback.trainer.state.log_history[-1]["step"], 2)

    def test_recursive_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cycle.yaml"
            p.write_text("extends: cycle.yaml")
            with self.assertRaises(ValueError):
                read_config(p)

    def test_physbrain_conversion_preserves_weights_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "original", Path(tmp) / "compatible"
            src.mkdir()
            config = {"model_type": "qwen3_vl", "text_config": {"rope_parameters": {
                "rope_type": "default", "rope_theta": 5000000,
                "mrope_section": [24, 20, 20], "mrope_interleaved": True}}}
            original = json.dumps(config)
            (src / "config.json").write_text(original)
            (src / "model.safetensors").write_bytes(b"unchanged")
            tokens = {"extra_special_tokens": ["<action>", "</action>"], "additional_special_tokens": ["<action>"]}
            (src / "tokenizer_config.json").write_text(json.dumps(tokens))
            prepare(src, dst)
            tokenizer = json.loads((dst / "tokenizer_config.json").read_text())
            self.assertEqual(tokenizer["additional_special_tokens"], ["<action>", "</action>"])
            self.assertNotIn("extra_special_tokens", tokenizer)
            self.assertEqual(json.loads((src / "tokenizer_config.json").read_text()), tokens)
            self.assertFalse((dst / "tokenizer_config.json").is_symlink())
            converted = json.loads((dst / "config.json").read_text())["text_config"]
            self.assertEqual(converted["rope_theta"], 5000000)
            self.assertEqual(converted["rope_scaling"]["mrope_section"], [24, 20, 20])
            self.assertEqual((src / "config.json").read_text(), original)
            self.assertTrue((dst / "model.safetensors").is_symlink())
            self.assertEqual((dst / "model.safetensors").read_bytes(), b"unchanged")
            with self.assertRaises(FileExistsError):
                prepare(src, dst)

    def test_non_qwen_backbone_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "original"
            src.mkdir()
            (src / "config.json").write_text('{"model_type":"other"}')
            with self.assertRaises(ValueError):
                prepare(src, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()


class WandbPreflightTest(unittest.TestCase):
    def test_transport_retry_queries_only_target_project(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import requests
        cfg = {"deployment": {"wandb_entity": "team"}, "training": {"wandb_project": "target"}}
        api = Mock(viewer=True)
        api.project.side_effect = [requests.exceptions.ReadTimeout(), SimpleNamespace(id="id")]
        factory, sleep = Mock(return_value=api), Mock()
        load_example("preflight").check_wandb_access(cfg, api_factory=factory, sleep=sleep)
        self.assertEqual(factory.call_count, 2)
        factory.assert_called_with(timeout=90)
        api.project.assert_called_with("target", entity="team")
        api.projects.assert_not_called()
        sleep.assert_called_once_with(5)

    def test_timeout_stops_after_configured_attempts(self):
        from unittest.mock import Mock
        import requests
        cfg = {"deployment": {"wandb_entity": "team", "wandb_check_attempts": 2},
               "training": {"wandb_project": "target"}}
        factory = Mock(side_effect=requests.exceptions.ReadTimeout())
        with self.assertRaisesRegex(RuntimeError, "network check failed"):
            load_example("preflight").check_wandb_access(cfg, api_factory=factory, sleep=Mock())
        self.assertEqual(factory.call_count, 2)

    def test_authentication_error_is_not_retried(self):
        from unittest.mock import Mock
        cfg = {"deployment": {"wandb_entity": "team"}, "training": {"wandb_project": "target"}}
        api = Mock(viewer=False)
        factory, sleep = Mock(return_value=api), Mock()
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            load_example("preflight").check_wandb_access(cfg, api_factory=factory, sleep=sleep)
        factory.assert_called_once()
        api.project.assert_not_called()
        sleep.assert_not_called()
