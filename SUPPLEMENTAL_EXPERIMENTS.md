# 单卡补实验：本地复现指南

状态：实现了协议、检查器和运行入口；CPU 测试不等于 CUDA 验证或正式实验结果。以下 GPU 命令在 4060 或 V100 的 Linux 主机上执行，不在 Mac 上测 GPU。每个作业只使用一张卡。

## 精度策略：历史 v1 与严格 FP32 v2 分开

`--precision-policy legacy_cudnn_tf32` 为默认值，保留 `supplement-v1` 的 CUDA matmul TF32=false、cuDNN TF32=true。新增 `--precision-policy strict_fp32` 使用 `supplement-v2`，两者均关闭 TF32，并统一用于 eager、graph、compile。`highest` matmul 设置和原 rtol=1e-4、atol=1e-5 不变。这里的严格 FP32 不意味着不同执行路径必须逐位相同，仍需逐条件正确性验证。

增加该设置的原因是实际诊断发现 MobileNetV3 在旧配置下有可复现的 eager/compile 差异，关闭 cuDNN TF32 后在已测输入上通过原门槛；不是通过放宽容差让失败消失。精度选择进入 task ID、manifest、gate 匹配和汇总，禁止把 v1 的 pass 或计时用于 v2。旧结果保持原样。新真实模型实验可显式选择 v2；补查历史数据的条件则保留对应精度配置。改变精度后必须重测同环境下所有参与比较的 action，不从旧表借用 eager 时间。

## 1. 真实模型与短任务资格

代码支持三个预训练图像分类模型：

| 模型 | 固定权重 | 覆盖目的 |
|---|---|---|
| ResNet-50 | `ResNet50_Weights.IMAGENET1K_V2` | 常规卷积推理 |
| ViT-B/16 | `ViT_B_16_Weights.IMAGENET1K_V1` | Attention/MLP 为主的视觉 Transformer |
| MobileNetV3-Large | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` | 较轻量的卷积推理，作为扩展候选 |

**模型本身没有“短任务认证”。** 本协议研究固定预训练模型完成有限批次评估后退出。默认 N={10,50,100,500} 仅是待审视的长度网格；尤其 ViT 的长条件需先测 eager 实际时长。使用 `characterize` 独立记录初始化、有效执行与总完成时间，在查看优化收益前，依据研究的时间范围冻结模型、batch、共同 horizon 和纳入／排除依据。不能以“不足以摊销 compile”循环定义短任务。若 500 步超出研究范围，所有平台使用同一调整后的核心网格；保留排除记录，可单列 longer-horizon sensitivity。

默认候选模型为 ResNet-50 与 ViT-B/16；可以显式加入 MobileNetV3-Large。两模型每平台有 120 个性能重复与 48 个 correctness 检查；三模型为 180+72。每个纳入正式证据的模型都以四平台同协议验证为目标，模型数量与覆盖/重复预算一起决定。当前入口不自动将任一模型或新结果认证为论文正式证据。

模型、权重与预处理使用 [TorchVision 官方模型接口](https://docs.pytorch.org/vision/0.27/models.html)，固定版本而非 `DEFAULT`。原 PyTorch 2.12.0+cu126 环境保留；可选依赖为与之匹配的 TorchVision 0.27.0+cu126。[版本对应表](https://github.com/pytorch/vision#installation)

## 2. 安装与 CPU 验证

在仓库根目录运行：

```bash
uv sync --frozen --extra real-models
OMP_NUM_THREADS=1 uv run --frozen --extra real-models python -m unittest discover -s tests
uv run --frozen --extra real-models python src/run_supplement.py --help
```

`real-models` extra 不安装时，模型专用测试会明确跳过；正式完整 CPU 验证使用上述命令。CPU 模型测试用随机权重/张量 fixture，绝不作为真实模型实验数据。

## 3. 先检查计划：默认不运行 GPU

```bash
bash scripts/run_local_supplement.sh 4060 core > /tmp/supplement-core-plan.json
bash scripts/run_local_supplement.sh v100 real --models resnet50 vit_b_16 > /tmp/supplement-real-plan.json
```

core 每平台 175 个检查：152 个定向 candidate，17 个 eager/best_eager 基线控制，6 个动态/不规则 graph 拒绝控制。四平台对应 608 个 candidate，基线/拒绝控制另计。主要矩阵为 8 families × 2 graph actions × 4 batches × 2 seeds；还有训练、独立 job 与 8-job reuse 检查。500-step 前缀与独立 N-step job 分开记录。

所有正式执行必须显式添加 `--execute`。程序拒绝 0 张或多张可见 CUDA 卡，并核对 `--platform` 与实际 GPU 型号。V100 所在主机若有多卡，选定其中一张；不会启动分布式训练或并发 GPU 作业。

## 4. 4060 / V100 correctness

在相应 GPU 主机的 Bash 中运行。选择实际分配给本实验的物理卡，以下 0 仅为单卡主机示例：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_supplement.sh 4060 core \
  --run-dir data/raw/supplement_4060_session01_core --execute

CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_supplement.sh v100 core \
  --run-dir data/raw/supplement_v100_session01_core --execute
```

V100 当前是否可用由实际资源决定；不要运行在别的卡上却标为 V100。失败是实验结果的一部分：命令会保存检查结果并继续其它条件，最后存在非预期失败会返回非零；必须查看 `summary.json`、逐条件 row 和 logs，而不是只看 shell 返回码。

## 5. 准备真实图像和固定权重

准备一个公开图像数据集的本地目录，例如 ImageNet 验证集的固定子集或 Imagenette。为跨机器复现记录精确版本／来源；不要把个人照片、模型权重或数据集提交到 Git。`prepare` 读取按相对路径排序的前 64 张图像，保存每张图像的 SHA-256，按固定权重自带的预处理生成 CPU 输入池，并下载、保存预训练权重。

```bash
uv run --frozen --extra real-models python src/run_supplement.py prepare \
  --images /path/to/public-dataset-images \
  --source 'Dataset name, version/split and public source URL' \
  --models resnet50 vit_b_16 mobilenet_v3_large \
  --out data/prepared/supplement_images_v1
```

`/path/to/...` 和 source 需替换为真实数据来源。不会静默退化成随机图片/随机权重。准备失败或参数改变时使用新的输出目录；不覆盖已冻结 artifacts。将整个 prepared 目录复制到其它平台，使用相同 manifest 和文件 hash。资产完整性检查及模型下载发生在计时外，正式 worker 不发起网络下载。

输入池按固定顺序循环，seed 固定起始 batch；因此 N 次执行未必代表 N 个不重复样本。报告图像池大小与重复使用方式。默认 batch=1；更大 batch 至少保留两个不同输入 batch，且四平台使用共同可运行的值。

## 6. 先测 eager 时长，再执行正式比较

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_supplement.sh 4060 characterize \
  --precision-policy strict_fp32 \
  --models resnet50 vit_b_16 mobilenet_v3_large \
  --assets data/prepared/supplement_images_v1 \
  --run-dir data/raw/supplement_4060_session01_characterize --execute
```

characterize 只运行 eager 数值检查和计时。查看 `summary.json` 的 `eager_duration_observations`，在其它平台补对应时长检查后冻结共同任务规模。代码不设置一个未经论证的秒数阈值，也不根据优化输赢裁剪任务。

冻结后运行同一矩阵的三条路径：相同 inference mode 的 eager、每步真实输入复制的 graph、compile-only。示例仍为候选默认网格，正式 horizon 若调整，四平台同步传入 `--lengths`：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_supplement.sh 4060 real \
  --precision-policy strict_fp32 \
  --models resnet50 vit_b_16 --batch 1 --lengths 10 50 100 500 --repeats 5 \
  --assets data/prepared/supplement_images_v1 \
  --run-dir data/raw/supplement_4060_session01_real --execute

CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_supplement.sh v100 real \
  --precision-policy strict_fp32 \
  --models resnet50 vit_b_16 --batch 1 --lengths 10 50 100 500 --repeats 5 \
  --assets data/prepared/supplement_images_v1 \
  --run-dir data/raw/supplement_v100_session01_real --execute
```

每个 action/batch/horizon 的两个 correctness seeds 均须通过，才准入该条件的性能重复。失败记为 `blocked`，不会冒充 eager 的性能或填成 0 秒。性能重复使用固定输入 seed=42、独立进程和独立冷编译缓存；它们测量时间变异，不冒充更多独立数据样本。

## 7. 计时、正确性与旧实验的边界

- 历史 runner 与历史 raw 不修改；补实验按 precision policy 标为 `supplement-v1` 或 `supplement-v2`，不能直接混入旧 schema 的汇总脚本，也不能合并不同精度的 repeats。
- 真实模型的任务时间从 CUDA context 已建立后开始，包括准备资产的 CPU 读取、模型实例化、权重加载/H2D、输入池准备、compile/capture/setup、N 次实际计算以及每步相同的输入 H2D。单独报告 initialization/setup/execution/total。
- 磁盘图像解码和预处理属于提前准备，网络下载、Python import、CUDA context 创建、hash 检查及 correctness inspection 不进入该任务时间。hash 检查可能预热文件缓存；不宣称冷磁盘加载或完整应用冷启动。
- 普通 compile 的首次编译在第一次执行中发生，因此 `execution_s` 包含该成本；`optimization_setup_s` 不是完整编译成本估计，也不要据此声称 steady-state step speedup。
- 性能循环不逐步额外 synchronize；相同输入复制逻辑用于所有 action，整个测量区间结束时同步。CPU 端选择下一批输入的开销包含在执行时间中。
- 新训练 checker 在 warm-up/capture 后将参数/buffers 原地恢复到初始状态，再比较同一 N 次实际更新的 loss、参数、梯度和 buffers。捕获可执行性及 CUDA 上的正确性须由真实 GPU 结果建立；CPU 单测只能验证比较器、快照和错误检测逻辑。
- 每个检查点保存独立快照，逐键报告误差与 first failing checkpoint。保留原 FP32 的 rtol=1e-4、atol=1e-5，失败不放宽阈值。
- core 的 `graphs_only` 是固定输入轨迹，`graphs_input_copy` 是匹配更新输入轨迹；core 本身只做 correctness，不用不同输入语义计算 speedup。

## 8. 保存、续跑与环境隔离

每次运行保存 `manifest.json`、`environment.json`、`rows/<task_id>.json`、`logs/<task_id>.log` 和 `summary.json`。manifest 绑定代码内容 hash、任务矩阵、资产 hash、GPU/CPU/runtime 环境身份。动态温度等只作诊断快照。

重复同一条命令会跳过已经记录的终态（含失败）并继续未完成条件；不会重抽失败样本直到通过。代码、主机、资产或矩阵变化时必须换新的 run directory。修复后的新实验与旧失败记录分别保留。不要手工删除失败 row。

每条件默认 900 秒 timeout；可用 `--timeout` 在运行前统一设定。超时是截尾/失败记录，不是 900 秒的有效性能数据。`--max-wall-seconds` 是两条任务之间检查的软预算，当前任务仍可能运行到单条件 timeout。中断/超时会终止该实验自己的进程组，包括 compiler workers；已原子保存的记录保留。未完成条件续跑使用新的冷缓存目录。

一次只在选定 GPU 上运行一个 campaign。run directory 有单写入锁；它不代替机器调度器的 GPU 资源分配。环境日志中的 cgroup root 字段不保证是进程最终生效配额，保留 membership 和原始记录供核查；不可读字段不猜测。

新租主机即使 GPU 型号相同也单列环境批次，同环境重测 baseline 与候选；不混合新旧时间或 queue 服务表。当前 local wrapper 仅开放 4060/V100；相同 Python 协议保留 A100/H100 platform 标识以便后续明确的云端运行。
