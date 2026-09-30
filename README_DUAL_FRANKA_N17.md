# 双臂 Franka · GR00T / PhysBrain

## 目录与环境

| 用途 | 路径 |
|---|---|
| 本机代码 | `/home/deepcybo/szl_ws/Isaac-GR00T` |
| 本机环境 | `/home/deepcybo/anaconda3/envs/gr00t` |
| 开发机代码 | `/user/users/szl/szl_ws/Isaac-GR00T` |
| 开发机环境 | `/user/users/szl/envs/gr00t` |
| 本机正式权重 | `checkpoints/dual_franka_full_50k` |

本机环境是uv虚拟环境，入口位于conda目录中，使用下面的activate；权重和环境大文件通过现有链接存于挂载盘。开发机Python、CUDA工具链、FFmpeg和编译工具在`/user/users/szl/envs/`持久保存，由`examples/dual_franka/env.sh`设置。

全量50000步已完成，batch64，耗时约16小时23分钟；最终权重已下载并通过SHA256与本机推理验收。自动巡检/开训已暂停，LoRA和freeze由用户提交任务。本仓库不保证任务提交之后的实时状态，需查看平台/W&B。

## 本机推理

两个终端分别执行：
```bash
cd /home/deepcybo/szl_ws/Isaac-GR00T
source /home/deepcybo/anaconda3/envs/gr00t/bin/activate
unset PYTHONPATH LD_PRELOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
```
终端1启动服务，等待`Server ready`：
```bash
python -m examples.dual_franka.run --config examples/dual_franka/server.yaml
```
终端2启动客户端：
```bash
python -m examples.dual_franka.inference_dual_ee14 --config examples/dual_franka/client.yaml
```

`server.yaml`引用`server_full.yaml`，默认监听`127.0.0.1:5555`。修改服务配置中的`model_path`并重启即可切换checkpoint；LoRA权重准备好后可用`server_lora.yaml`，目前不要指向尚未下载的模型。

`client.yaml`默认`runtime.mode: replay`、`execute: false`，读取保存的观测，不初始化相机/机器人。日志自动保存在`deployment_records/client_runs/时间戳/`，包括原始模型动作和执行修正后的命令。显式指定`runtime.output`时不会覆盖已有目录。

### 真机联动

先启动原有双Franka RPC服务，核对`client.yaml`中的机器人IP/端口、三路相机序列号及task，然后设置：
```yaml
runtime:
  mode: robot
  execute: true
  iterations: 100
  output: null
  output_root: deployment_records/client_runs
```
再次运行同一客户端命令。开始时两臂home 5秒后打开夹爪；正常结束或Ctrl+C时也会home并打开夹爪。初始reset失败不自动重试。服务超时、非法动作或硬件拒绝命令会中止，不复用旧动作。

沿用旧RPC及夹爪反馈转换。动作40帧中默认执行前10帧，30Hz；同步推理存在等待空档。夹爪关闭阈值、速度/力、位姿限幅位于`execution`中，沿用旧client设置。RGB原图424×240由服务处理成256×256，客户端不重复resize或归一化。

已验证：4090离线加载、直接调用、ZeroMQ和客户端无动作回放，预热约47–51ms/请求，输出40×14有限动作。未以这些检查代替真实机器人任务成功率验证。报告见`deployment_records/local_inference_ready_20260928.json`。

## 提交训练任务

每个任务分配4张GPU，挂载路径保持`/user/users/szl`，平台通过密钥注入`WANDB_API_KEY`。以下命令以前台进程运行，不需要nohup；同一组卡上的任务应排队。

LoRA VLM＋全量AE：
```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit.sh examples/dual_franka/train_lora.yaml
```
冻结VLM＋全量AE：
```bash
bash /user/users/szl/szl_ws/Isaac-GR00T/examples/dual_franka/submit.sh examples/dual_franka/train_freeze.yaml
```
full模式对应`train_full.yaml`，已完成的实验不要用相同目录从头重跑。恢复对应任务时将文件名改为`train_lora_resume.yaml`、`train_freeze_resume.yaml`或`train_full_resume.yaml`。必须有该实验的完整checkpoint；模式和参数范围不能在resume时更换。

所有模式从原始PhysBrain前16层＋新随机AE独立开始，默认50000步、有效batch64=4卡×16、累积1，每10000步保存，保留最近2个训练checkpoint。LoRA/freeze不继承full最终权重。新实验应修改output_dir和wandb_id。

训练基础参数集中在`train_job.yaml`，模式配置覆盖具体实验参数。`submit.sh`必须传YAML；已删除无参数自动运行旧烟测的分支。本机无4卡训练数据，训练入口供开发机任务容器使用。

### 可训练参数与W&B

`model.vlm_mode`可选full/lora/freeze；`ae_trainable: true`使AE全量训练。LoRA默认为语言attention q/k/v/o，rank16、alpha32、dropout0.05，视觉和原始VLM冻结。`lora_layers: null`表示全部保留层。

`trainable_include`可使用完整参数名glob缩小当前模式的训练范围；`trainable_exclude`优先排除。过滤器不能将freeze参数重新解冻。参数匹配错误或最终无可训练参数会报错；实际清单在训练输出的`trainable_parameters.json`。

W&B项目：`shenzhaolong1330-beihang-university/dual-franka-gr00t-n17`。预检查只查询目标项目，网络超时90秒，最多3次；训练初始化等待180秒。认证失败不当成网络重试，训练强制online。新任务容器不会自动继承旧容器的登录；网络仍失败需检查该任务到api.wandb.ai:443的出网/代理。Triton的NFS缓存警告不等于训练失败。

## 数据契约

对外state/action都是14D，顺序为左EE 6、右EE 6、左右夹爪各1。state为位置、旋转向量和反馈宽度；action为EE平移/旋转向量增量与夹爪绝对宽度。单位米/弧度，不重排、不二值化、不再次差分；旋转增量沿用RPC左乘约定。内部原生补齐到132D并使用mask，对外仍为14D。

数据语义为`heuristic_closed_plateau_zero_v1`，夹爪关闭稳定段由旧标签启发式重标注，不能视作恢复了原始遥操作命令。LeRobot v3转换为独立v2.1副本，保留固定1885训练/99验证episode划分；不覆盖源数据。完整40帧动作窗口不跨episode，`allow_padding: false`。

仅训练集计算原生分位数归一化统计，验证和推理复用。服务负责state归一化、图像处理和action反归一化；checkpoint保存processor与统计，不复用OpenPI/StarVLA统计。task按任务索引读取，图像顺序head、left_wrist、right_wrist。

## 导出与验证

远端完成训练后：
```bash
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
python -m examples.dual_franka.export --config examples/dual_franka/export_full.yaml
```
LoRA用`export_lora.yaml`。其他checkpoint修改导出YAML的checkpoint/output路径。输出含完整权重、架构、processor、统计、EE14契约与SHA256 manifest；不依赖Cosmos下载。LoRA adapter保留在原生模型中，无需单独指定adapter。

下载到本机临时目录后，用`verify_assets.py --manifest <目录>/manifest.json --root <目录> --report <报告路径>`校验再发布。保留训练完整checkpoint用于恢复，不能拿推理包恢复optimizer。

无硬件直接/服务回放分别使用`replay_full_direct.yaml`和`replay_full_remote.yaml`，入口：
```bash
python -m examples.dual_franka.replay --config examples/dual_franka/replay_full_direct.yaml
```
LoRA有对应的`replay_lora_*`配置。远端服务可通过`ssh -N -L 5555:127.0.0.1:5555 -p 30295 dev@10.30.13.100`转发；本地与隧道不要占用同一个端口。

## 保留文件与清理范围

- `run.py / preflight.py / submit.sh / env.sh / training_support.py`：训练与提交入口。
- `inference_dual_ee14.py / robot_control.py / dual_franka_robotiq_rpc_client.py`：策略客户端与原机器人控制。
- `modality.py / prepare_dataset.py / verify_dataset.py / prepare_physbrain.py`：字段注册、转换和数据准备。
- `export.py / replay.py / verify_assets.py / check_environment.py / probe.py`：导出及部署检查。
- `deployment_records/client_runs`：用户运行日志；`data_validation/replay_fixture.npz`：默认回放必需样本；正式验收报告、来源哈希与SSH主机指纹保留。

2026-09-28只清理本机：删除旧烟测配置、一次性模式检查、未使用Cosmos下载脚本、3个本次新增临时测试文件、旧烟测日志和缓存。上游原生测试保留。训练基础配置合并前后，三模式及恢复配置逐项相同。权重、环境、数据、真机日志、核心实现均保留；开发机未同步此次清理。删除清单：`deployment_records/cleanup_20260928.json`。

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
