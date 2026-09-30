"""EE14 stored labels: do not difference them a second time."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

KEYS = ["left_ee", "right_ee", "left_gripper", "right_gripper"]
CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["head", "left_wrist", "right_wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(40)), modality_keys=KEYS,
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF, format=ActionFormat.DEFAULT) for _ in KEYS]),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}
register_modality_config(CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
