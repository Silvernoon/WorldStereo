# WorldStereo WAM

WorldStereo 的 WAM (World Action Model) 风格封装，模仿 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 的项目结构，提供统一的训练/推理入口、Hydra 配置管理和可安装的 Python 包。

## 项目结构

```text
WorldStereo/
├── pyproject.toml                    # 可安装包配置 (pip install -e .)
├── __init__.py
├── configs/
│   ├── train.yaml                    # 训练主配置
│   ├── inference.yaml                # 推理主配置
│   ├── model/
│   │   └── worldstereo.yaml          # 模型配置
│   ├── data/                         # 数据集配置 (与 FastWAM 一致)
│   │   ├── libero_2cam.yaml          # LIBERO 双相机
│   │   └── robotwin.yaml             # RoboTwin 三相机
│   └── task/                         # 训练任务配置
│       ├── libero_2cam224_1e-4.yaml
│       └── robotwin_3cam384_1e-4.yaml
├── scripts/
│   ├── train.py                      # Hydra 训练入口
│   ├── inference.py                  # Hydra 推理入口
│   ├── train_zero1.sh                # DeepSpeed ZeRO-1 训练脚本
│   ├── accelerate_configs/
│   │   └── accelerate_zero1.yaml
│   └── ds_configs/
│       └── ds_zero1_config.json
├── src/
│   └── worldstereo_wam/              # WAM 核心包
│       ├── __init__.py
│       ├── runtime.py                # 统一 API: create_worldstereo, run_inference, run_training
│       ├── trainer.py                # 训练器 (类似 FastWAM Wan22Trainer)
│       ├── checkpoint_compat.py      # FastWAM ckpt 兼容性检查工具
│       ├── datasets/                 # LeRobot 数据管线 (移植自 FastWAM)
│       │   └── lerobot/
│       │       ├── robot_video_dataset.py
│       │       ├── base_lerobot_dataset.py
│       │       ├── processors/
│       │       └── transforms/
│       └── utils/
│           ├── config_resolvers.py   # Hydra/OmegaConf 自定义 resolver
│           ├── fs.py
│           ├── logging_config.py
│           ├── misc.py
│           ├── pytorch_utils.py
│           └── samplers.py
├── models/                           # 原 WorldStereo 模型实现
├── src/                              # 原 WorldStereo 工具函数
├── run_camera_control.py             # 原入口脚本 (仍可用)
└── run_multi_traj.py                 # 原入口脚本 (仍可用)
```

## 安装

```bash
# 创建 conda 环境
conda create -n worldstereo python=3.10 -y
conda activate worldstereo

# 安装 PyTorch (根据你的 CUDA 版本)
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# 安装 WorldStereo WAM 包
pip install -e .
```

## 推理

### 使用 Hydra 配置入口 (推荐)

```bash
# Camera control (单视图)
python scripts/inference.py task_type=camera_control input_path=examples/images

# 多轨迹全景
python scripts/inference.py task_type=panorama input_path=examples/panorama

# 3D 重建
python scripts/inference.py task_type=reconstruction input_path=examples/reconstruction

# 切换模型类型
python scripts/inference.py model_type=worldstereo-memory-dmd
python scripts/inference.py model_type=worldstereo-memory
python scripts/inference.py model_type=worldstereo-camera

# 启用 W8A8 量化 (节省 ~50% 显存)
python scripts/inference.py w8a8=true w8a8_save_path=quantized/transformer.pt
```

### 多卡分布式推理

```bash
torchrun --nproc_per_node=8 scripts/inference.py \
    task_type=panorama \
    input_path=examples/panorama \
    fsdp=true
```

### 使用 Python API

```python
from worldstereo_wam import create_worldstereo

# 加载模型
worldstereo = create_worldstereo(
    model_path="hanshanxue/WorldStereo",
    model_type="worldstereo-memory-dmd",
    device="cuda",
    quantize_w8a8=False,
)

# 调用 pipeline
output = worldstereo.pipeline(**pipeline_kwargs)
```

### 使用原脚本 (仍然支持)

```bash
# 原 camera control 脚本
torchrun --nproc_per_node=1 run_camera_control.py \
    --model_type worldstereo-camera \
    --input_path examples/images \
    --output_path outputs

# 原 multi-trajectory 脚本
torchrun --nproc_per_node=8 run_multi_traj.py \
    --model_type worldstereo-memory-dmd \
    --task_type panorama \
    --input_path examples/panorama \
    --output_path outputs
```

## 训练

### 使用 Hydra 配置入口

```bash
# 基本训练
python scripts/train.py

# 覆盖配置
python scripts/train.py learning_rate=1e-5 batch_size=4 num_epochs=20

# 使用 wandb 日志
python scripts/train.py wandb.enabled=true wandb.project=my-project
```

### DeepSpeed ZeRO-1 分布式训练

```bash
bash scripts/train_zero1.sh
```

## 配置系统

WorldStereo WAM 使用 [Hydra](https://hydra.cc/) 进行配置管理，支持：

- **配置组合**: 通过 `defaults` 组合多个配置文件
- **命令行覆盖**: 直接在命令行覆盖任意配置项
- **多运行**: 使用 `--multirun` 进行超参搜索

### 主要配置文件

| 文件 | 用途 |
|------|------|
| `configs/train.yaml` | 训练超参、优化器、调度器等 |
| `configs/inference.yaml` | 推理参数、输入输出路径等 |
| `configs/model/worldstereo.yaml` | 模型架构配置 |

## 与 FastWAM 的对应关系

| FastWAM | WorldStereo WAM |
|---------|-----------------|
| `fastwam.runtime.create_fastwam()` | `worldstereo_wam.runtime.create_worldstereo()` |
| `fastwam.runtime.run_training()` | `worldstereo_wam.runtime.run_training()` |
| `fastwam.trainer.Wan22Trainer` | `worldstereo_wam.trainer.WorldStereoTrainer` |
| `scripts/train.py` | `scripts/train.py` |
| `configs/train.yaml` | `configs/train.yaml` |
| `configs/model/fastwam.yaml` | `configs/model/worldstereo.yaml` |

## License

请参阅 [LICENSE.md](LICENSE.md)。

## 数据集 (与 FastWAM 一致)

WorldStereo WAM 直接移植了 FastWAM 的 LeRobot 数据管线，位于 `src/worldstereo_wam/datasets/`，支持 LIBERO 与 RoboTwin。

### 数据配置

| 配置 | 说明 |
|------|------|
| `configs/data/libero_2cam.yaml` | LIBERO 双相机 (agentview + wrist)，水平拼接，224 分辨率 |
| `configs/data/robotwin.yaml` | RoboTwin 三相机，384 分辨率 |

### 数据目录结构

与 FastWAM 相同，需要 LeRobot 格式数据集：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### 在 Python 中直接使用 dataset

```python
from worldstereo_wam import RobotVideoDataset

dataset = RobotVideoDataset(
    dataset_dirs=["./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"],
    shape_meta=shape_meta,        # 见 configs/data/libero_2cam.yaml
    num_frames=33,
    video_size=[224, 448],
    is_training_set=True,
)
sample = dataset[0]
# sample: { video, action, proprio, prompt, context, ... }
```

> 注意：文本 embedding 缓存需要先用 FastWAM 的 `scripts/precompute_text_embeds.py`
> 逻辑生成，或将 `text_embedding_cache_dir` 指向已有缓存。

## 能否复用 FastWAM 的 checkpoint？

**不能直接整包加载。** 两者核心架构不同：

| | FastWAM | WorldStereo |
|---|---------|-------------|
| 基座 | Wan2.2-TI2V-5B | WanTransformer3DModel |
| 结构 | MoT (video expert + action expert) | DiT + ControlNet |
| 额外模块 | ActionDiT / proprio / action head / StereoEncoder | 相机控制 / GGM / SSM 几何记忆 |
| 输出 | 视频 + 动作 | 多视角一致视频 → 3D 重建 |

因此 `strict=True` 的整包加载是不可能的。可行的只是 **部分迁移**：两者都基于 Wan DiT，
部分 transformer block 的同名同形状权重理论上可以作为初始化迁移（不保证效果）。

为此提供了 `checkpoint_compat` 工具：

```python
import torch
from worldstereo_wam import (
    create_worldstereo,
    inspect_compatibility,
    extract_loadable_subset,
)

# 1. 加载 WorldStereo 模型
model = create_worldstereo(model_type="worldstereo-camera", device="cuda")
transformer = model.pipeline.transformer

# 2. 检查 FastWAM ckpt 与 WorldStereo transformer 的兼容性
report = inspect_compatibility("path/to/fastwam_checkpoint.pt", transformer)
print(report.summary())
# 输出: 名称+形状匹配 / 形状不匹配 / 缺失 / 多余 的参数统计

# 3. 如果有可迁移的子集，抽取并以 strict=False 加载
loadable = extract_loadable_subset("path/to/fastwam_checkpoint.pt", transformer)
if loadable:
    transformer.load_state_dict(loadable, strict=False)
```

实际中大概率会看到 "No transplantable weights found"，因为两者参数命名空间几乎不重叠。
**推荐做法**：仍使用 WorldStereo 官方权重 (`hanshanxue/WorldStereo`)，
FastWAM ckpt 仅作为研究性初始化实验。

