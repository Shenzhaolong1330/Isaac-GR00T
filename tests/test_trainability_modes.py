"""Training-policy regression tests on a tiny Qwen-shaped module tree."""

import copy
from types import SimpleNamespace
import unittest

from gr00t.model.modules.trainability import configure_trainability, keep_frozen_modules_eval
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(self, name, nn.Linear(8, 8))

    def forward(self, x):
        return sum(getattr(self, n)(x) for n in ["q_proj", "k_proj", "v_proj", "o_proj"])


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"self_attn": Attention()}) for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer["self_attn"](x)
        return x


class Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.2))
        self.model.language_model = Language()

    @property
    def visual(self):
        return self.model.visual

    @property
    def language_model(self):
        return self.model.language_model

    def forward(self, x):
        return self.language_model(self.visual(x))


class Tiny(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.backbone = nn.Module()
        self.backbone.model = Qwen()
        self.action_head = nn.Linear(8, 2)
        configure_trainability(self, cfg)

    def forward(self, x):
        return self.action_head(self.backbone.model(x))

    def train(self, mode=True):
        super().train(mode)
        keep_frozen_modules_eval(self)
        return self


def config(mode, **overrides):
    data = dict(
        vlm_mode=mode,
        ae_trainable=True,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_layers=None,
        select_layer=2,
        trainable_include=None,
        trainable_exclude=[],
    )
    data.update(overrides)
    return SimpleNamespace(**data)


class TrainabilityTest(unittest.TestCase):
    def test_modes_update_only_selected_parameters_and_reload(self):
        for mode in ["full", "lora", "freeze"]:
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                m = Tiny(config(mode)).train()
                before = {n: p.detach().clone() for n, p in m.named_parameters()}
                optimizer = torch.optim.AdamW(
                    [p for p in m.parameters() if p.requires_grad], lr=0.01
                )
                x = torch.randn(3, 8)
                m(x).square().mean().backward()
                for n, p in m.named_parameters():
                    if not p.requires_grad:
                        self.assertIsNone(p.grad, n)
                optimizer.step()
                changed = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
                self.assertTrue(any(n.startswith("action_head") for n in changed))
                for n, p in m.named_parameters():
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(p, before[n]), n)
                if mode == "lora":
                    self.assertTrue(any(".lora_B." in n for n in changed))
                    self.assertTrue(
                        all(
                            ".lora_" in n
                            for n, p in m.backbone.named_parameters()
                            if p.requires_grad
                        )
                    )
                    self.assertEqual(len(m.trainability_report["lora_modules"]), 8)
                if mode == "freeze":
                    self.assertFalse(m.backbone.training)
                    self.assertFalse(any(n.startswith("backbone") for n in changed))
                if mode != "full":
                    self.assertFalse(m.backbone.model.visual.training)
                m.eval()
                clone = Tiny(copy.deepcopy(m.config))
                clone.load_state_dict(m.state_dict(), strict=True)
                clone.eval()
                torch.testing.assert_close(m(x), clone(x), rtol=0, atol=0)

    def test_filters_and_layer_selection(self):
        m = Tiny(config("full", trainable_include=["action_head.*"], trainable_exclude=["*.bias"]))
        self.assertEqual(m.trainability_report["trainable_names"], ["action_head.weight"])
        m = Tiny(config("lora", lora_layers=[1], lora_target_modules=["q_proj"]))
        self.assertEqual(
            m.trainability_report["lora_modules"],
            ["model.language_model.layers.1.self_attn.q_proj"],
        )
        m = Tiny(config("full", ae_trainable=False))
        self.assertFalse(any(p.requires_grad for p in m.action_head.parameters()))

    def test_invalid_configs_fail_closed(self):
        for kwargs in [
            dict(vlm_mode="bad"),
            dict(trainable_include=["missing.*"]),
            dict(trainable_exclude=["missing.*"]),
            dict(vlm_mode="freeze", ae_trainable=False),
            dict(vlm_mode="lora", lora_target_modules=["typo"]),
            dict(vlm_mode="lora", lora_layers=[2]),
            dict(vlm_mode="lora", lora_rank=0),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mode = kwargs.pop("vlm_mode", "full")
                Tiny(config(mode, **kwargs))


if __name__ == "__main__":
    unittest.main()
