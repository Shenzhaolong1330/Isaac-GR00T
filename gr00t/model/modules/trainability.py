"""Explicit GR00T training policies. Adapters are part of native full checkpoints."""

from fnmatch import fnmatchcase

import torch


POLICY_FIELDS = (
    "vlm_mode",
    "ae_trainable",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "lora_layers",
    "trainable_include",
    "trainable_exclude",
)


def configure_trainability(model, config):
    mode = config.vlm_mode
    if mode is None:
        if config.trainable_include is not None or config.trainable_exclude:
            raise ValueError("Parameter filters require explicit vlm_mode")
        return
    if mode not in {"full", "lora", "freeze"}:
        raise ValueError(f"Unknown vlm_mode: {mode}")
    if not isinstance(config.ae_trainable, bool):
        raise ValueError("ae_trainable must be a boolean")
    backbone = model.backbone
    backbone.requires_grad_(mode == "full")
    model.action_head.requires_grad_(config.ae_trainable)
    matched = []
    if mode == "lora":
        from peft import LoraConfig, inject_adapter_in_model

        if config.lora_rank <= 0 or config.lora_alpha <= 0 or not 0 <= config.lora_dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0,1)")
        targets = config.lora_target_modules
        if (
            not targets
            or not isinstance(targets, list)
            or not all(isinstance(x, str) for x in targets)
        ):
            raise ValueError("lora_target_modules must be a nonempty list")
        layers = config.lora_layers
        if layers is not None and (
            not layers
            or any(type(i) is not int or not 0 <= i < config.select_layer for i in layers)
        ):
            raise ValueError("lora_layers must select existing retained language layers")
        found = set()
        for name, module in backbone.model.named_modules():
            if ".language_model.layers." not in "." + name or ".self_attn." not in name:
                continue
            leaf = name.rsplit(".", 1)[-1]
            layer = int(name.split("language_model.layers.", 1)[1].split(".", 1)[0])
            if leaf in targets and (layers is None or layer in layers):
                if not isinstance(module, torch.nn.Linear):
                    raise ValueError(f"LoRA target is not Linear: {name}")
                matched.append(name)
                found.add(leaf)
        if found != set(targets):
            raise ValueError(f"Unmatched LoRA targets: {set(targets) - found}")
        adapter = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=matched,
            bias="none",
            init_lora_weights=True,
        )
        # In-place injection preserves Qwen's forward API and module hierarchy.
        backbone.model = inject_adapter_in_model(adapter, backbone.model)
    params = dict(model.named_parameters())
    for patterns in (config.trainable_include, config.trainable_exclude):
        if patterns is None:
            continue
        if not isinstance(patterns, list) or not all(isinstance(x, str) for x in patterns):
            raise ValueError("Parameter filters must be lists of glob patterns")
        for pattern in patterns:
            if not any(fnmatchcase(name, pattern) for name in params):
                raise ValueError(f"Unmatched parameter pattern: {pattern}")
    for name, parameter in params.items():
        allowed = config.trainable_include is None or any(
            fnmatchcase(name, p) for p in config.trainable_include
        )
        excluded = any(fnmatchcase(name, p) for p in config.trainable_exclude)
        parameter.requires_grad_(parameter.requires_grad and allowed and not excluded)
    if not any(p.requires_grad for p in params.values()):
        raise ValueError("Training policy selects zero trainable parameters")
    backbone.tune_llm = any(p.requires_grad for p in backbone.model.language_model.parameters())
    backbone.tune_visual = any(p.requires_grad for p in backbone.model.visual.parameters())
    # Keep native dropout/eval controls consistent with the explicit parameter mask.
    head = model.action_head
    if hasattr(head, "set_trainable_parameters"):
        head.tune_projector = any(
            p.requires_grad
            for name in ("state_encoder", "action_encoder", "action_decoder", "position_embedding")
            for p in getattr(head, name, torch.nn.Identity()).parameters()
        )
        head.tune_diffusion_model = any(p.requires_grad for p in head.model.parameters())
        head.tune_vlln = any(
            p.requires_grad
            for module in (head.vlln, head.vl_self_attention)
            for p in module.parameters()
        )
    model.trainability_report = {
        "mode": mode,
        "lora_modules": matched,
        "total": sum(p.numel() for p in params.values()),
        "trainable": sum(p.numel() for p in params.values() if p.requires_grad),
        "trainable_names": [n for n, p in params.items() if p.requires_grad],
        "frozen_names": [n for n, p in params.items() if not p.requires_grad],
    }


def keep_frozen_modules_eval(model):
    """Called after train(): frozen VLM blocks must not reactivate dropout."""
    if model.config.vlm_mode is None or not model.training:
        return
    backbone = model.backbone
    candidates = [
        backbone,
        backbone.model.visual,
        backbone.model.language_model,
        *backbone.model.language_model.layers,
        model.action_head,
        *model.action_head.children(),
    ]
    for module in candidates:
        if not any(p.requires_grad for p in module.parameters()):
            module.eval()
