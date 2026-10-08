# 双臂 Franka GR00T：训练与部署

Python入口和提交脚本仍在 `examples/dual_franka/`，YAML统一放在其下的 `configs/`。以下命令从仓库根目录执行。本机为 `/home/deepcybo/szl_ws/Isaac-GR00T`，开发机为 `/user/users/szl/szl_ws/Isaac-GR00T`。

## 配置分类

| 配置/目录 | 用途 |
|---|---|
| `configs/deploy/server.yaml` | 唯一的当前模型服务配置，切换权重只改 `model_path` |
| `configs/deploy/client_replay.yaml` | 保存观测回放，不连接机器人或相机 |
| `configs/deploy/client_robot.yaml` | 实时采集、机器人RPC和动作执行 |
| `configs/replay/` | 模型直接调用/网络调用的重复性、耗时检查 |
| `configs/train/` | 训练、恢复及被继承的基础配置 |
| `configs/export.yaml` | 唯一导出配置，设置checkpoint、VLM和输出目录 |
| `support/` | 入口自动加载的机器人控制、RPC、modality和训练回调 |
| `tools/` | 数据/模型准备、校验和任务环境预检 |

旧顶层YAML链接、归档、一次性检查和重复提交脚本已删除，使用下文的新路径。当前模型仍为 `checkpoints/dual_franka_full_40k`。训练配置展开后的参数、输出路径、W&B ID保持不变。

## 本机启动

每个终端先执行：

```bash
cd /home/deepcybo/szl_ws/Isaac-GR00T
source /home/deepcybo/anaconda3/envs/gr00t/bin/activate
unset PYTHONPATH LD_PRELOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
```

终端1启动模型服务：

```bash
python -m examples.dual_franka.run --config examples/dual_franka/configs/deploy/server.yaml
```

终端2用保存的同一份观测请求5次推理，不执行动作：

```bash
python -m examples.dual_franka.inference_dual_ee14 --config examples/dual_franka/configs/deploy/client_replay.yaml
```

需要真机联动时，先启动已有的双Franka机器人RPC服务，然后使用独立实机配置：

```bash
python -m examples.dual_franka.inference_dual_ee14 --config examples/dual_franka/configs/deploy/client_robot.yaml
```

**实机命令会让机器人运动。** 配置为 `mode: robot`、`execute: true`、100次循环。机器人IP、相机序列号、任务和执行参数都在该文件中。沿用原控制逻辑：连接策略/相机/RPC → home 5秒并打开夹爪 → 推理/执行 → 正常结束或Ctrl+C后再次home并打开夹爪。初始reset失败不自动重试。

每次预测40帧，按30Hz执行前10帧，再重新观测；同步推理期间有等待空档。夹爪阈值0.03m、速度0.3、力10及原动作限幅不变。两个客户端配置均启用 `timestamp_output: true`，日志分别写到 `deployment_records/client_replay/<时间戳>/` 和 `deployment_records/robot_runs/<时间戳>/`；`steps.jsonl`记录模型输出、执行命令和耗时，重复运行无需改目录。

## 切换权重与远端服务

修改 `configs/deploy/server.yaml` 的 `model_path` 并重启server，客户端配置共用。填写已下载、校验完成的推理导出目录，例如：

| 实验 | model_path |
|---|---|
| 原全量4万步（当前） | `checkpoints/dual_franka_full_40k` |
| 对齐π优化器、全量batch128、10万步 | `checkpoints/dual_franka_full_pi_aligned_b128_100k` |
| 原全量batch64、10万步 | `checkpoints/dual_franka_full_100k` |
| 原全量5万步 | `checkpoints/dual_franka_full_50k` |
| LoRA 5万步 | `checkpoints/dual_franka_lora_50k` |
| freeze 5万步 | `checkpoints/dual_franka_freeze_50k` |
| 原版Qwen3-VL对照5万步 | `checkpoints/dual_franka_qwen3vl_full_50k` |

远端server也默认监听 `127.0.0.1:5555`；在本机开隧道后，客户端仍连接localhost：

```bash
ssh -N -L 5555:127.0.0.1:5555 -p 30295 dev@10.30.13.100
```

本机模型服务与隧道不能同时占用5555。相机和机器人始终由本机客户端连接。

## 数据和模型边界

客户端发送三路原始RGB uint8图像（424×240）、物理14D state和任务文本。服务使用checkpoint内的processor完成256×256补边、state归一化、预测和action反归一化，返回40×14物理动作。客户端不resize、不再次归一化或差分。

顺序为左EE6、右EE6、左/右夹爪宽度；state为绝对位姿和反馈宽度，action为EE增量和夹爪宽度，单位米/弧度。夹爪标签保留 `heuristic_closed_plateau_zero_v1`，旋转沿用原RPC的左乘约定。

## 训练、恢复与导出

平台任务挂载 `/user/users/szl`，分配4卡，通过密钥注入 `WANDB_API_KEY`。独立LoRA训练的提交命令：

```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit.sh examples/dual_franka/configs/train/train_lora.yaml
```

| 实验 | configs/train/ 下的配置 |
|---|---|
| PhysBrain16全量VLM＋AE，5万步/batch64 | `train_full.yaml` |
| 独立LoRA VLM＋全量AE，5万步/batch64 | `train_lora.yaml` |
| 独立冻结VLM＋全量AE，5万步/batch64 | `train_freeze.yaml` |
| 原全量5万步续训到10万步，保留optimizer并重启LR | `train_full_100k.yaml` |
| 原版Qwen3-VL前16层＋新AE，5万步/batch64 | `train_qwen3vl_full.yaml` |
| 对齐π优化器，全量10万步/batch128 | `train_full_pi_aligned.yaml` |

恢复使用同名 `_resume.yaml`，保持output_dir、W&B ID、训练模式及优化参数。原生恢复模型、optimizer、scheduler、step和RNG；数据采样游标不保证精确恢复。新实验同时修改output_dir与W&B ID。

公共设置已合并到 `configs/train/base.yaml`，原Cosmos/PhysBrain/1000步短测模板已删除。训练统一使用 `submit.sh <实验YAML>`，不带参数只打印用法，不会自动启动短测或第二阶段。原来三个专用提交脚本已删除，按上表选择对应YAML即可。

开发机导出示例：

```bash
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
python -m examples.dual_franka.export --config examples/dual_franka/configs/export.yaml
```

导出前修改 `configs/export.yaml` 的 `checkpoint`、`vlm`、`output` 三项；当前示例对应对齐π优化器的10万步实验。原版Qwen3-VL实验应使用自己的VLM目录，不能套用PhysBrain。训练恢复使用完整训练checkpoint；部署使用包含模型、processor、统计和契约的推理包，已有输出目录不会覆盖。

## 模型回放检查

直接加载模型，不需要server；自动读取当前server配置的模型路径：

```bash
python -m examples.dual_franka.replay --config examples/dual_franka/configs/replay/direct.yaml
```

检查已经启动的服务：

```bash
python -m examples.dual_franka.replay --config examples/dual_franka/configs/replay/remote.yaml
```

这两种检查均不执行机器人动作。remote配置不再保留无效的checkpoint字段。结果写入 `deployment_records/replay_direct/` 或 `deployment_records/replay_remote/`，重复检查会更新同目录结果。

历次运行产物仍保留在 `deployment_records/`，checkpoint和数据目录未修改。过期的过程文档已删除，日常启动以上述入口为准。整理后已完成32项离线测试，并用full 4万步权重验证本机直接调用、ZeroMQ服务和回放客户端：动作形状40×14，固定种子结果一致，4090预热后约47ms/次，峰值模型显存约4.7GiB。未启动新的训练或执行机器人动作。
