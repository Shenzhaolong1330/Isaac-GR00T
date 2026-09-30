# 双臂 Franka GR00T：PhysBrain16 训练与部署

## 当前交付（2026-09-26）

已完成4卡20步全量训练/恢复测试，以及full、LoRA、freeze各4步真实训练与checkpoint加载回放。**正式目标为5万步，由用户选择模式后提交任务启动**。三种模式都使用PhysBrain前16层，随机初始化原生GR00T AE；不是完整28层VLM，也不继承Cosmos动作头。Cosmos不作为依赖。

仓库来自 `Shenzhaolong1330/Isaac-GR00T`，基线 `51d4c89`，分支 `codex/dual-franka-ee14`。当前实例修改在工作区，尚未提交推送。

## 提交任务：直接使用这个命令

任务必须挂载 `/user/users/szl`，分配**单机4张GPU**，使用Linux x86_64及支持当前GPU/PyTorch cu128的NVIDIA驱动。容器需要能访问W&B。

在任务平台的密钥环境变量中设置 `WANDB_API_KEY`。不要将密钥写入YAML、Git或命令行。当前开发容器的登录已验证，但新任务容器不一定共享它的登录文件。

```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit.sh examples/dual_franka/train_lora.yaml
```

上例选择LoRA＋全量AE，full/freeze只替换配置文件名。传入YAML时脚本执行预检后直接运行该配置；不传参数才执行旧的500→1000步烟测。它调用原生GR00T Trainer，不建立另一套训练框架。没有启动50k任务。

正式配置为 `examples/dual_franka/train_full.yaml`、`train_lora.yaml`、`train_freeze.yaml`；恢复用对应的 `_resume.yaml`。旧的 `train_job.yaml` / `train_job_resume.yaml` 仅用于1000步烟测。首次运行前可修改 `training.output_dir` 和 `deployment.wandb_id`；**新实验两项一起改，恢复时两项保持不变**。不要把其他实验的输出目录用于新训练。

50k配置默认：有效batch64（4×16×1）、BF16、LR 1e-4、warmup1000步、cosine、clip1、每卡2个worker、动作40帧、每1000步独立验证32个观测、每10000步保存完整checkpoint。若显存不足，修改global_batch_size为32且gradient_accumulation_steps为2，或16和4；这里global_batch_size是累积前的跨卡batch，不是单卡batch。

只检查、不训练：
```bash
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
python -m examples.dual_franka.preflight --config examples/dual_franka/train_lora.yaml
```

任务中断后，从最新完整checkpoint继续（不要重新运行首次启动脚本）：
```bash
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
python -m examples.dual_franka.run --config examples/dual_franka/train_lora_resume.yaml
```

完整checkpoint由原生Trainer保存optimizer、scheduler、step及RNG。当前原生配置ignore_data_skip=true，不承诺恢复到完全相同的采样游标。10→20步完整恢复已验证；500→1000将在正式任务中执行。

W&B：`shenzhaolong1330-beihang-university/dual-franka-gr00t-n17`。同一run ID恢复，强制online；记录flow loss、物理单位误差、左右夹爪MAE、旋转角误差、梯度/首次权重变化、学习率、吞吐和显存。小规模验证不能代表实机成功率，不以loss=0.001验收。

## 持久环境

| 内容 | 开发机路径 |
|---|---|
| 仓库 | `/user/users/szl/szl_ws/Isaac-GR00T` |
| Python包环境 | `/user/users/szl/envs/gr00t` |
| Python3.12.13、uv、独立CUDA12.9、FFmpeg6运行库 | `/user/users/szl/envs/gr00t-support` |
| Git、GCC/G++14、make、ninja、转换用FFmpeg命令 | `/user/users/szl/envs/gr00t-runtime` |
| 运行/编译缓存 | `/user/users/szl/cache/gr00t-*` |
| 模型 | 仓库 `checkpoints/PhysBrain1.5-2B-gr00t`，链接目标也位于挂载盘 |

`source examples/dual_franka/env.sh` 设置所有路径，不依赖旧容器的 `/root/.local/share/uv/python`。不需要conda activate，不修改StarVLA/OpenPI环境。宿主NVIDIA驱动由任务平台提供，不能装进Python环境代替。

核心依赖：Python3.12.13、torch2.9.0+cu128、transformers4.57.3、torchcodec0.8、flash-attn2.8.3、DeepSpeed0.17.6；本轮用SDPA。依赖清单与运行报告保存于deployment_records。环境只为当前Linux架构准备，不是任意操作系统的可移植安装包。

本机仓库 `/home/deepcybo/szl_ws/Isaac-GR00T`；环境 `/home/deepcybo/anaconda3/envs/gr00t` 指向已授权挂载盘。开发机env.sh不用于本机，本机用：
```bash
cd /home/deepcybo/szl_ws/Isaac-GR00T
source /home/deepcybo/anaconda3/envs/gr00t/bin/activate
unset PYTHONPATH LD_PRELOAD
export CUDA_HOME=/usr/local/cuda
export HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
```

## 数据与统计

源数据 `insert_tube_rack_all_E1984_gripper_relabel_v01` 保持不变；独立v2.1副本在 `/user/users/szl/dataset/dual_franka/gr00t_ee14_v21`。

| 划分 | episode | 原始帧 | 完整40帧窗口起点 | 排除尾部起点 |
|---|---:|---:|---:|---:|
| train | 1885 | 1574917 | 1501620 | 73297 |
| validation | 99 | 82834 | 79002 | 3832 |

沿用原有划分，哈希 `a127b6d237f05f4e172effea06a067e4948a7decf5d55669615055009c159267`。`prepare_dataset.py` 直接读取v3 Parquet与共享视频时间偏移，独立输出逐episode文件；数值字段逐列回读校验，episode索引映射到各划分本地编号，保留来源编号。视频重新编码为lossless H264，不能声称与源文件字节相同。`verify_dataset.py` 校验代表性来源/任务的起、中、末帧并生成无硬件回放样本。

固定顺序：左EE6、右EE6、左夹爪1、右夹爪1。state是物理位置/旋转向量和夹爪反馈；action是逐步EE平移/旋转向量增量和夹爪米制宽度。ABSOLUTE仅表示直接使用存储标签，不额外差分、不重排、不二值化。夹爪语义保留 `heuristic_closed_plateau_zero_v1`，它不是原始命令的精确恢复。

`modality.py` 映射 head/left_wrist/right_wrist，任务按task_index读取。allow_padding=false，40帧窗口不跨episode。原生采样器不保证严格无放回，因此处理样本数不直接称为完整遍历次数。

原生统计只在训练集原始帧拟合，文件 `train/meta/stats.json`；验证集复制同一统计。模型用原生q01/q99归一化和裁剪，不复用OpenPI/StarVLA统计。processor内部补到132维，mask只有14维有效；对外仍是14D。

三路原始RGB由同一个GR00T processor等比例补边至256×256，关闭随机增强。真实batch的Qwen grid为每路[1,16,16]。客户端不重复resize或归一化；服务处理state归一化及action反归一化。

## 模型兼容与部署（训练完成后）

`prepare_physbrain.py` 生成兼容视图，转换Transformers5风格RoPE及tokenizer特殊token配置为锁定4.57.3可读格式，原始权重不变。实测可训练参数：视觉约407M、保留语言约1151M、原生AE约878M。

完成1000步后，编辑export.local.yaml的checkpoint/output路径，导出不依赖原始PhysBrain目录的推理包：
```bash
python -m examples.dual_franka.export --config examples/dual_franka/export.local.yaml
python -m examples.dual_franka.run --config examples/dual_franka/server_physbrain.yaml
```

服务使用原生ZeroMQ+MessagePack NumPy，默认127.0.0.1:5555。切换checkpoint只改服务YAML的model_path并重启。远端隧道：
```bash
ssh -p 30295 -N -L 5555:127.0.0.1:5555 dev@10.30.13.100
```

回放客户端（另一个终端），配置direct或remote：
```bash
python -m examples.dual_franka.replay --config examples/dual_franka/replay_remote.yaml
```

回放返回40×14物理动作，**不加载机器人SDK、不连接相机或夹爪、不执行home/reset/动作**。还没有训练好的本实例checkpoint；训练后独立导出与两端加载仍需验收，不能把随机初始化动作头用于机器人。

## 修改与验证范围

- 新增双臂modality、配置、数据转换/视频核验、任务预检、原生Trainer验证回调、推理导出与回放。
- 原生experiment.run增加可选Trainer设置回调；本地Qwen路径识别与嵌入结构配置支持离线构建；策略增加可选固定随机种子以核验回放。
- 原始数据、旧checkpoint、其他项目环境不改。数据、权重、日志、凭据不进入Git。
- 已验证：两端核心导入和CUDA，转换数值字段，54组代表性视频边界，真实样本前向/反向、14D mask、三类模块非零梯度、有限推理输出。报告在deployment_records。
- 已追加验证：4卡有效batch64真实训练20步，10步保存后恢复至20步，三类模块权重更新，W&B同一run连续记录；6项回归测试通过。
- 5万步全量任务已启动；最终模型质量、最终导出和实机效果仍待验证。

## 真实短训验收

测试目录 `/user/users/szl/runs/gr00t_n17/physbrain16_b64_pretest_v2_20260926`，保留checkpoint-10和checkpoint-20，正式任务不从它初始化。短测只缩短总步数/调度并增加验证保存频率，模型更新范围和4×16有效batch不变。

[W&B测试记录](https://wandb.ai/shenzhaolong1330-beihang-university/dual-franka-gr00t-n17/runs/franka16-pretest-v2-20260926)。第20步训练loss=1.168；20步仅验证训练链路，不能评判收敛或机器人成功率。原生Trainer恢复后打印的累计train_loss分母包含恢复前步数，不作为该阶段平均loss，应查看逐步train/loss。

显存实测每卡约36–40GB（nvidia-smi），已缓存数据时约1.1–1.3秒/optimizer步；首批缓存、初始化、验证和checkpoint写盘额外计时，不能直接用短测步速保证长训总耗时。

测试发现并修复rank0单独调用Trainer.log造成控制状态不同步的停顿：自定义指标改为直接写入W&B和log_history，不触发Trainer的on_log控制回调。第2步记录真实参数更新，避开warmup首步LR=0。另直接比较原始权重与checkpoint的第1和第16层attention权重，两者均有非零更新。

详细验收报告：deployment_records/training_pretest_result.json。该短训已结束；后续用户授权的full 50k已单独启动，不从短训权重继续。

## VLM full / LoRA / freeze（配置入口）

三种模式都使用PhysBrain前16层，AE默认全量训练。模式在原生模型构建时应用，早于DeepSpeed和optimizer创建；不添加另一套Trainer，不修改数据字段、统计和推理协议。

| 配置 | VLM更新范围 | AE | 正式步数/有效batch |
|---|---|---|---|
| examples/dual_franka/train_full.yaml | 保留的语言、视觉与连接模块全部更新 | 全量 | 50000 / 64 |
| examples/dual_franka/train_lora.yaml | 语言attention q/k/v/o LoRA，基础权重和视觉模块冻结 | 全量 | 50000 / 64 |
| examples/dual_franka/train_freeze.yaml | 完全冻结，冻结模块保持eval模式 | 全量 | 50000 / 64 |

2026-09-26已启动full 50k；LoRA等待full完成、权重下载及本机推理验证后启动，freeze没有启动。新实验请修改output_dir与wandb_id；三份默认配置已使用不同目录和ID。启动LoRA示例（full/freeze只替换文件名）：
```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit.sh examples/dual_franka/train_lora.yaml
```
传入YAML后，只执行该配置的训练，不再执行旧的500→1000步两阶段流程。不传参数仍为原来的1000步烟测。也可以source env.sh后直接运行：
```bash
python -m examples.dual_franka.run --config examples/dual_franka/train_lora.yaml
```
恢复使用对应的train_lora_resume.yaml（full/freeze同理）。同一实验恢复时不要修改模式、LoRA结构、参数白名单/黑名单；入口会检查checkpoint配置并拒绝不匹配。训练范围切换必须作为新实验，不能复用旧optimizer状态。start_from_checkpoint也要求训练策略字段一致；本次三份正式配置均从原始PhysBrain＋新AE开始。

### 指定可训练参数

```yaml
model:
  vlm_mode: lora            # full / lora / freeze
  ae_trainable: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
  lora_layers: null         # null=全部保留层；[12,13,14,15]=最后4层（索引从0开始）
  trainable_include: null   # null=不额外限制
  trainable_exclude: []
```

顺序是：先按模式/ae_trainable选中参数，再与include取交集，最后应用exclude。exclude优先；过滤器只能缩小当前模式的可训练范围，不能把freeze模式下的VLM重新解冻。想训练指定原始VLM参数，应选择full再筛选。

例如只训练第16层语言主干和整个AE：
```yaml
model:
  vlm_mode: full
  ae_trainable: true
  trainable_include:
    - backbone.model.model.language_model.layers.15.*
    - action_head.*
  trainable_exclude: []
```

例如LoRA＋AE，但冻结state encoder：
```yaml
model:
  vlm_mode: lora
  ae_trainable: true
  trainable_exclude:
    - action_head.state_encoder.*
```

参数匹配使用完整名称的glob（*通配）；LoRA目标则是语言self_attn下的Linear模块名，不会顺带命中视觉attention。未匹配名称、越界层号、非法模式或最终零个可训练参数均报错。每次运行生成trainable_parameters.json，包含实际LoRA模块和完整可训练/冻结参数清单；parameter_policy.json汇总分组数量；first_update.json记录梯度与实际权重变化。

vlm_mode为null时保留上游tune_*旧行为；显式指定模式时，以新模式和参数筛选为准。无需同时修改tune_llm/tune_visual等旧字段。

### 保存和测试说明

使用已锁定的PEFT 0.17.1在原模型中注入LoRA，保留Qwen原始forward接口。原生GR00T checkpoint包含基础VLM、AE及LoRA参数，配置记录adapter结构；不需要额外下载或单独指定adapter文件。推理按checkpoint配置自动重建，LoRA目前保留为adapter，不做自动合并。不能删除或手改checkpoint里的vlm_mode/lora_*字段。

tests/test_trainability_modes.py覆盖模式更新范围、冻结参数不变、LoRA层选择、白名单/黑名单、非法配置及严格保存加载。真实4卡短测使用test_full.local.yaml、test_lora.local.yaml、test_freeze.local.yaml，每种4步、有效batch64，输出位于/user/users/szl/runs/gr00t_n17/modes_*_b64_20260926；这些是验收模型，不用于机器人任务，也不作为正式50k的初始化。

保存加载和无动作回放入口：
```bash
python -m examples.dual_franka.check_mode_checkpoints --config examples/dual_franka/test_modes_reload.local.yaml
```
它重新加载checkpoint，对所有命名参数抽样核对保存值，并对freeze/LoRA的基础VLM逐张量抽样核对原始权重；随后固定种子重复预测，检查物理动作形状[1,40,14]及有限值。不会导入机器人SDK或执行动作。报告在deployment_records/modes_reload_result.json。

### 三模式实测结果（2026-09-26）

| 模式 | 可训练参数 | rank0 PyTorch峰值分配 | 检查 |
|---|---:|---:|---|
| full | 2,435,326,080 | 30.06GiB | 视觉/语言/AE更新 |
| lora | 881,417,344（其中LoRA3,670,016） | 14.01GiB | 基础VLM冻结，LoRA/AE更新 |
| freeze | 877,747,328 | 8.64GiB | VLM冻结，AE更新 |

峰值分配不包含CUDA上下文和缓存保留，不能直接等同nvidia-smi占用。每种模式均完成4步有效batch64，保存完整checkpoint、在线上传W&B，再独立加载并生成[1,40,14]有限物理动作；固定种子重复输出差异为0。冻结模式/LoRA基础VLM各检查493个参数张量的抽样值，均保持原值。9项配置/模式测试与本次改动的ruff检查通过。

W&B：[full](https://wandb.ai/shenzhaolong1330-beihang-university/dual-franka-gr00t-n17/runs/franka16-modes-full-20260926)、[LoRA](https://wandb.ai/shenzhaolong1330-beihang-university/dual-franka-gr00t-n17/runs/franka16-modes-lora-20260926)、[freeze](https://wandb.ai/shenzhaolong1330-beihang-university/dual-franka-gr00t-n17/runs/franka16-modes-freeze-20260926)。报告：deployment_records/modes_acceptance.json。

这些短测只验证训练范围与链路，不用4步loss排名选择策略，也不代表5万步收敛效果。测试进程已全部退出。


## 本次两轮长训和巡检（2026-09-26）

顺序固定：full 50000步 → 导出/下载/本机推理验收 → **原始PhysBrain＋全新AE**的LoRA 50000步。两轮有效batch均64、AE全量训练；第二轮不继承第一轮权重或optimizer。LoRA只更新语言attention的q/k/v/o适配器，视觉与基础VLM冻结。每10000步保存，当前最多保留2个训练checkpoint，最终导出独立保留。

- [full W&B](https://wandb.ai/shenzhaolong1330-beihang-university/dual-franka-gr00t-n17/runs/franka16-full-ae-full-b64-50k) 已在线运行。
- LoRA预定runID：`franka16-lora-ae-full-b64-50k`，未启动前没有训练记录。
- 远端日志：`deployment_records/full50k.log`；结束状态：`full50k.exit`、`full50k.ended`；对应LoRA为`lora50k.*`。
- 已设置当前Codex任务每2小时巡检（automation `gr00t`），检查进度/错误/W&B/显卡并执行后续阶段。保持本机可用和Codex任务可运行；远端训练本身通过nohup脱离终端运行，本机关闭不会停止已启动的训练，但会推迟本机下载验收及第二轮调度。
- 阶段记录：`deployment_records/full_lora_sequence.json`。不要再手动启动同名实验或第二个训练进程。

早期稳定步速约1.2秒，full 5万步约17小时量级，加上首次缓存、验证和保存；只是估计。LoRA耗时以后续实测为准。

远端完成后用原生checkpoint导出离线包：
```bash
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
python -m examples.dual_franka.export --config examples/dual_franka/export_full.yaml
# 第二轮完成后替换为 export_lora.yaml
```
对应本机目录`checkpoints/dual_franka_full_50k`、`checkpoints/dual_franka_lora_50k`位于现有挂载盘链接。下载先进入`.partial`目录，逐文件核对manifest.json的SHA256后再发布；不复制W&B凭据，不覆盖旧权重。

`dual_franka_full_smoke4`仅为4步部署验收权重，不能用来判断任务效果，也不要用于实机任务。

## 本机开始策略服务、再连接机器人

机器人侧沿用之前常用的`dual_franka_robotiq_rpc_client.py`，已从Le-nero复制到`examples/dual_franka/`，内容不改。`robot_control.py`沿用StarVLA/OpenPI的状态转换、夹爪命令、限幅及reset顺序；仅增加RPC明确拒绝时停止的检查。没有依赖StarVLA模型包或vla_runtime。

本机每个终端先执行：
```bash
cd /home/deepcybo/szl_ws/Isaac-GR00T
source /home/deepcybo/anaconda3/envs/gr00t/bin/activate
unset PYTHONPATH LD_PRELOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
```
不要沿用ROS的PYTHONPATH。机器人可选依赖已装在这个独立环境，版本见`examples/dual_franka/requirements-robot.txt`，不需要安装另一套LeRobot。

**终端1：模型服务。** 最终权重下载验收后：
```bash
python -m examples.dual_franka.run --config examples/dual_franka/server_full.yaml
```
切换LoRA用`server_lora.yaml`。服务YAML的`model_path`决定checkpoint，无需客户端再选择full/LoRA。地址默认`127.0.0.1:5555`。首次编译/加载慢于后续请求。

**终端2：客户端。** `examples/dual_franka/client.yaml`默认`runtime.mode: replay`、`execute: false`，只读保存的样本，不导入或初始化机器人、夹爪和相机：
```bash
python -m examples.dual_franka.inference_dual_ee14 --config examples/dual_franka/client.yaml
```
每次修改`runtime.output`为新的日志目录（已存在会报错）。`runtime.iterations`为请求次数。

需要真实机器人联动时，在该YAML里明确设置：
```yaml
runtime:
  mode: robot
  execute: true
  iterations: 100
  output: deployment_records/robot_run_001
```
核对同文件的机器人`172.16.0.1:4242`、三路相机序列号和task文本，先启动既有双Franka机器人RPC服务，再启动上述客户端。流程是：检查策略接口 → 打开相机/RPC → **两臂home 5秒后打开夹爪** → 采集/推理/执行 → 正常完成或Ctrl+C时再次home并打开夹爪 → 关闭资源。初始reset失败直接退出，不重试运动；退出reset若失败会明确报错。

动作仍为左EE增量6、右EE增量6、左/右夹爪米制宽度，旋转向量采用原RPC的左乘约定。每次返回40帧，仅执行前10帧、30Hz，然后重新观测。夹爪关闭阈值0.03m、速度0.3、力10、位置/旋转逐维限幅0.3m/0.12rad沿用旧client配置，可在execution中调整；不要将限幅解释为模型单位换算。模型物理输出与执行修正后的命令分别写入steps.jsonl。同步推理存在等待空档，不是持续30Hz闭环。

GR00T和旧StarVLA不同：客户端发送三路原始**RGB uint8 424×240**和物理state；GR00T服务内processor完成256×256补边、state归一化及action反归一化。客户端不resize、不归一化、不重新差分。native ZeroMQ+MessagePack NumPy，连接时校验三路图像、state/action键和40帧horizon；该原生握手没有提供checkpoint统计哈希，需通过服务配置确认所用模型。请求超时/错误/非法shape或NaN直接终止当前会话，不复用旧动作。

远端推理时服务仍监听回环地址，本机先开隧道：
```bash
ssh -N -L 5555:127.0.0.1:5555 -p 30295 dev@10.30.13.100
```
本机和远端服务不要同时占用本机5555。客户端配置保持localhost即可；相机和机器人始终在本机。

本次只做模拟控制与无动作回放，未执行真实机器人运动。模型checkpoint可加载不等于实机任务成功。


### 下载校验及无动作验收命令

开发机持久化传输工具为`/user/users/szl/envs/gr00t-support/rsync/rsync`，带自身的系统库，未修改训练环境的库路径。以下在本机执行，LoRA将full替换为lora。先确认远端export已成功，最终目录不存在；不要把未完成checkpoint当作完成结果。
```bash
rsync -a --partial --info=progress2 \
  --rsync-path=/user/users/szl/envs/gr00t-support/rsync/rsync \
  -e 'ssh -p 30295 -o UserKnownHostsFile=/home/deepcybo/szl_ws/Isaac-GR00T/deployment_records/dev_known_hosts' \
  dev@10.30.13.100:/user/users/szl/szl_ws/Isaac-GR00T/checkpoints/dual_franka_full_50k/ \
  checkpoints/dual_franka_full_50k.partial/
python -m examples.dual_franka.verify_assets \
  --manifest checkpoints/dual_franka_full_50k.partial/manifest.json \
  --root checkpoints/dual_franka_full_50k.partial \
  --report deployment_records/full50k_download_verified.json
mv -T checkpoints/dual_franka_full_50k.partial checkpoints/dual_franka_full_50k
python -m examples.dual_franka.replay --config examples/dual_franka/replay_full_direct.yaml
```
启动server_full.yaml后，再执行replay_full_remote.yaml。结果保存actions.npy及result.json，比较同一seed的动作、有限值、[1,40,14]和延迟。上述回放不导入机器人模块。LoRA对应server_lora.yaml、replay_lora_direct.yaml、replay_lora_remote.yaml。


### 本机部署实测（2026-09-26）

已下载`checkpoints/dual_franka_full_smoke4`并逐文件通过SHA256。4090上离线直接加载、原生ZeroMQ服务、复用机器人client的无硬件回放均通过，固定seed=42的3次输出形状[1,40,14]、全有限，三条路径最大差异0。直接调用预热后约46ms，localhost服务约49ms；只对同一固定样本测得，不能当作完整相机/实机周期或一般p95。PyTorch峰值分配约4.66GiB，保留约4.73GiB，不包含全部CUDA/桌面占用。

客户端10项模拟测试通过，RPC文件与原文件SHA256一致。临时短测服务已停止，未连接相机或机器人，未执行home/reset。报告`deployment_records/local_smoke4_acceptance.json`。最终full/LoRA 50k权重完成后仍会重复下载校验和本机推理验收，不能用短测通过替代最终验收。

复现纯模拟测试（先unset PYTHONPATH）：
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dual_franka_robot_client.py
```


## 提交任务的 W&B 超时报错修复（2026-09-28）

已修复preflight.py中`list(api.projects(entity))`枚举团队所有项目导致的ReadTimeout。现在只访问配置指定的项目，并读取项目ID以实际发起检查；单次请求超时90秒、网络错误最多尝试3次、间隔5秒，认证/权限错误不作为网络超时重试。真正训练的W&B初始化超时180秒，仍强制online，不静默降级。配置在train_job.yaml的deployment下：wandb_timeout_seconds、wandb_check_attempts、wandb_init_timeout_seconds，LoRA/freeze均继承。

原提交命令不变。失败发生在预检查、未进入训练，重新提交仍使用train_lora.yaml或train_freeze.yaml，不需要resume配置。任务容器继续注入WANDB_API_KEY。若重试仍提示network check failed，应检查任务容器到api.wandb.ai:443的出网/代理，开发容器可连通不代表任务容器一定可连通。日志开头的DeepSpeed Triton NFS提示是缓存性能警告，与这次W&B请求超时不是同一问题。

验证：现有部署测试9项通过，远端两份YAML检查通过；开发容器无凭据且仅1张可见GPU，因此本轮没有冒充完成新任务容器的四卡训练或认证验收。

## 全量续训到10万步，以及原版Qwen对照（2026-09-28）

平台任务均挂载 `/user/users/szl`，申请4张GPU，在平台环境变量/密钥中注入 `WANDB_API_KEY`。下面是两次独立提交的启动命令，不需要手动激活环境，不要在同一组4卡上同时运行。

**PhysBrain：从原 full 的 checkpoint-50000 续训到累计100000步**

```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit_full_100k.sh
```

配置：`examples/dual_franka/train_full_100k.yaml`。恢复模型、Adam optimizer和累计步数；在50000步后单独重启一个50000步的学习率周期：前1000步warmup到1e-4，再cosine到0。不会使用原训练已经降为0的学习率继续空跑。新的输出是 `/user/users/szl/runs/gr00t_n17/physbrain16_full_ae_full_b64_100k`，不会覆盖原5万步实验。

**官方Qwen3-VL-2B-Instruct：独立全量训练50000步**

```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit_qwen3vl_full.sh
```

配置：`examples/dual_franka/train_qwen3vl_full.yaml`。官方模型版本固定为 `89644892e4d85e24eaac8bacfd4f463576704203`；基础权重在 `checkpoints/Qwen3-VL-2B-Instruct`，兼容配置视图在 `checkpoints/Qwen3-VL-2B-Instruct-gr00t`。同样保留前16层、全量训练VLM和新初始化的GR00T AE，不加载PhysBrain或任何已训练AE。输出是 `/user/users/szl/runs/gr00t_n17/qwen3vl2b16_full_ae_full_b64_50k`。

两轮保持原来的数据划分、图像处理、统计、40帧动作窗口、有效batch64（4卡×16）、BF16、AdamW、学习率1e-4和验证设置。每10000步保存，本次 `save_total_limit: 6`，保留本轮所有整万步checkpoint。W&B仍上传到 `shenzhaolong1330-beihang-university/dual-franka-gr00t-n17`，两轮使用不同run ID。

如果新任务中断，用新输出目录的最近checkpoint恢复：

```bash
cd /user/users/szl/szl_ws/Isaac-GR00T
bash examples/dual_franka/submit.sh examples/dual_franka/train_full_100k_resume.yaml
# 或：
bash examples/dual_franka/submit.sh examples/dual_franka/train_qwen3vl_full_resume.yaml
```

续训再次恢复按累计global step继续同一个新学习率周期，不重复warmup。沿用原来的 `ignore_data_skip` 数据恢复方式，不保证逐样本重放中断前的采样序列。脚本启动前会检查GPU数量、数据和W&B；没有在线凭据时明确失败，不偷偷转为offline。

代码只扩展原生checkpoint路径类型，并在原生恢复完成后替换学习率曲线，optimizer状态不清空。已用原生Transformers Trainer小模型验证Adam状态保留、二次中断恢复和scheduler保存加载；四卡完整checkpoint恢复仍需在提交的四卡任务上实际执行。

**本地4万步权重**：`checkpoints/dual_franka_full_40k` 是原全量实验checkpoint-40000的独立推理导出，包含模型、processor、统计和契约；训练optimizer保留在开发机原checkpoint目录。切换到它只需把服务YAML的 `model_path` 改为 `checkpoints/dual_franka_full_40k` 并重启服务。本地5万步权重继续保留。

本轮验收：官方Qwen权重两端SHA256一致；开发机真实样本前向/反向通过，视觉、语言、AE均有非零梯度；本地4万步导出SHA256通过，4090离线回放输出1×40×14，预热约48ms，固定种子重复差为0。正式四卡训练及在线W&B将在平台任务启动后执行。

新训练完成后，分别用 `export_full_100k.yaml` 或 `export_qwen3vl_full.yaml` 调用原有 `python -m examples.dual_franka.export --config examples/dual_franka/<文件名>`。Qwen导出必须使用自己的Qwen配置和tokenizer，不能套用PhysBrain导出配置。

2026-09-29 续训修复：Trainer管理LambdaLR时，DeepSpeed的lr_scheduler允许为空；只在双方存在不同的活动调度器时报错。原判断错误地拦截了合法恢复。已通过开发机单卡原生Trainer＋DeepSpeed ZeRO-2 BF16小模型测试，确认Adam状态逐值恢复、scheduler每次更新推进一步、二次恢复结果一致。测试不是四卡完整GR00T续训验收。继续使用 submit_full_100k.sh 提交；修复不修改原checkpoint、optimizer或训练配置。

## 对齐π0.5优化器的4卡对照实验（2026-09-30）

之前双臂π0.5的正式命令是全局batch128，4张B300每卡32、累计10万步。这次GR00T对照使用4张RTX PRO 6000 96GB，每卡32、累积1，有效batch128。沿用原始PhysBrain前16层＋新随机AE，全量训练VLM和AE；是独立实验，不继承此前5万/10万步GR00T权重或optimizer。

平台挂载 `/user/users/szl`、分配4卡，并通过平台密钥注入 `WANDB_API_KEY`。提交命令：

```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit_full_pi_aligned.sh
```

配置集中在 `examples/dual_franka/train_full_pi_aligned.yaml`：

| 参数 | 本次配置 |
|---|---|
| optimizer | PyTorch fused AdamW |
| beta1 / beta2 / epsilon | 0.9 / 0.95 / 1e-8 |
| weight decay | 1e-10，含bias、norm，范围与π相同 |
| 全局梯度裁剪 | 1.0 |
| 峰值学习率 | 1e-5 |
| warmup | 100 optimizer steps；第0步LR=1e-5/101 |
| cosine终点 | 第100000步，LR=1e-6 |
| batch | 4×32×累积1=128 |
| 保存/验证 | 每10000步保存，保留10个；每1000步固定32样本验证 |

这次是一条连续10万步曲线，第5万步不重启warmup。初始化和配置差异意味着仍不能把两模型的训练loss数值直接比较：GR00T继续使用自己的数据、归一化、40帧窗口和14D有效维mask。优化器更新规则及学习率曲线对齐，不宣称跨框架/跨GPU逐位等价。

训练输出：`/user/users/szl/runs/gr00t_n17/physbrain16_full_ae_full_pi_aligned_b128_100k`。
W&B项目：`shenzhaolong1330-beihang-university/dual-franka-gr00t-n17`；run ID：`franka16-full-ae-full-pi-aligned-b128-100k`。

中断后使用相同配置恢复最近checkpoint，保留optimizer和scheduler，不重新warmup：

```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit.sh examples/dual_franka/train_full_pi_aligned_resume.yaml
```

恢复检查会拒绝更换beta、学习率曲线或batch；改变这些条件应另开实验。训练完成后用 `export_full_pi_aligned.yaml` 调用现有export入口，输出 `checkpoints/dual_franka_full_pi_aligned_b128_100k`。

改动范围：TrainingConfig新增AdamW和OpenPI曲线选项，experiment透传AdamW参数，原生Gr00tTrainer按开关选择曲线及decay范围；旧配置默认行为保留。dual_franka入口的batch检查改为配置值，吞吐按实际有效batch计算。没有更换数据框架、模型或loss。

验证：CPU实际Optax/PyTorch曲线和AdamW更新对照通过，原生Trainer两次恢复及累积计步通过；开发机PRO6000真实数据、每卡batch32全量更新3步通过。开发机只有1张可见GPU，正式4卡NCCL和在线W&B由提交入口在任务容器内检查，通过后才训练。

补充验收：实际PRO6000上的原生Gr00tTrainer＋DeepSpeed ZeRO-2 BF16测试通过（2→4→6两次恢复），Adam moments及FP32主权重逐值恢复，beta/epsilon/decay和scheduler计步正确。真实batch32测试中视觉、语言、AE均检测到非零梯度和权重变化。报告位于deployment_records/pi_aligned_*_validation_20260930.json；临时3步测试的大权重已清理，正式模型保留。
