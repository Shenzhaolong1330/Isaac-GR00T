# 双臂 Franka GR00T

主目录只保留日常入口：

| 文件 | 用途 |
|---|---|
| `run.py` | 原生GR00T训练/模型服务入口 |
| `inference_dual_ee14.py` | 回放或真实机器人客户端 |
| `replay.py` | 无硬件模型检查 |
| `export.py` | 导出独立推理权重 |
| `submit.sh` | 唯一训练提交入口，必须指定实验YAML |
| `env.sh` | 开发机任务环境 |
| `requirements-robot.txt` | 本机机器人依赖 |

- **配置**：`configs/deploy/` 下是 [模型服务](configs/deploy/server.yaml)、[回放客户端](configs/deploy/client_replay.yaml)、[实机客户端](configs/deploy/client_robot.yaml)。训练在 `configs/train/`，导出只用 [export.yaml](configs/export.yaml)。
- **内部代码**：`support/` 包含机器人RPC、控制、数据modality和训练回调，入口会自动加载。
- **准备工具**：`tools/` 保留数据转换/校验、PhysBrain兼容处理、权重校验以及任务环境预检。按需使用 `python -m examples.dual_franka.tools.<工具名> --help`。

旧归档、一次性检查、Cosmos下载工具、重复提交脚本和兼容链接已删除；使用 [主说明](../../README_DUAL_FRANKA_N17.md) 中的新路径。数据、权重和运行记录仍在原位置。
