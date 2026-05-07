# BaGLM Online 实时任务进度判断

## 1. 基本原理

### 1.1 什么是 BaGLM

BaGLM (Bayesian Grounding with Large Multimodal Models) 是一个 **training-free** 的视频步骤定位方法。给定一个任务（如"补充可乐"）和该任务的步骤列表（如"从货架上拿可乐"、"把可乐放到另一个货架"），BaGLM 能够判断视频中**当前正在执行哪个步骤**。

### 1.2 核心架构

系统由三个独立模块组成：

```
相机帧 (N帧)                    LMM 推理 (GPU)               贝叶斯滤波 (CPU)
┌──────────┐    Socket     ┌─────────────────────┐    ┌───────────────────┐
│ 机器人相机 │ ───────────→ │ 模块1: VSG 推理      │    │ 模块3: 贝叶斯滤波  │
│ (RGB帧)   │              │ "当前在做什么？"      │    │                   │
│           │              │ → scores [S+1]       │    │ belief + 历史信息   │
│           │              ├─────────────────────┤    │ → 当前步骤判断      │
│           │              │ 模块2: Progress 推理  │    │                   │
│           │              │ "这个步骤进行到哪了？" │    │ 输出:              │
│           │              │ → scores [S, 10]     │    │  步骤名 + 置信度    │
└──────────┘              └─────────────────────┘    └───────────────────┘
```

- **VSG（Video Step Grounding）**：LMM 观察当前几帧图像，从步骤列表中选择最匹配的步骤（或"都不是"）
- **Progress**：LMM 判断每个步骤的执行进度（0-9 共 10 个等级）
- **贝叶斯滤波**：融合 VSG 和 Progress 的结果，结合步骤之间的先决依赖关系（如"拿可乐"必须在"放可乐"之前），维护一个 belief 概率分布

### 1.3 "Online" 是什么意思

BaGLM 本身就是为 online 设计的——VSG 和 Progress 推理只看当前几帧，不需要完整视频。历史信息通过贝叶斯滤波的 belief 状态累积。这意味着：

- 帧可以逐段实时获取（从相机）
- 每收到一段帧就能立即得到当前步骤判断
- 不需要等视频录制完成

### 1.4 通信架构

```
机器人端                              推理服务器 (GPU)
┌─────────────────┐    TCP Socket    ┌──────────────────────┐
│ 采集 N 帧图像     │ ─────────────→ │ 接收帧数据            │
│ (2秒 × 2fps = 4帧)│                │ LMM VSG + Progress    │
│                  │ ←───────────── │ 贝叶斯滤波更新        │
│ 执行动作控制       │   JSON 响应     │ 返回: 步骤 + 置信度   │
└─────────────────┘                 └──────────────────────┘
```

协议为简单的长度前缀二进制协议：
1. 客户端发送：4字节长度 + JSON header + 原始帧数据
2. 服务端回复：4字节长度 + JSON 响应

---

## 2. 环境配置

### 2.1 两个 Conda 环境

由于 InternVL 和 RoboBrain 依赖不同版本的 transformers，需要两个环境：

| 环境 | 模型 | transformers | 创建方式 |
|------|------|-------------|---------|
| `baglm` | InternVL2.5-8B | 4.49.0 | 按项目 README 原始安装 |
| `baglm-robobrain` | RoboBrain2.5-4B | >=4.57.0 | 克隆 baglm 后升级 transformers |

创建 RoboBrain 环境：

```bash
conda create -n baglm-robobrain --clone baglm -y
conda activate baglm-robobrain
pip install 'transformers>=4.57.0,<5.0'
```

### 2.2 环境变量

启动时需设置（已包含在启动脚本中）：

```bash
export HF_HOME="/mnt/public1/dais/hf_cache"   # 模型缓存目录
export HF_HUB_OFFLINE=1                         # 离线模式，不访问 huggingface.co
export LD_LIBRARY_PATH="$HOME/opt/ffmpeg-n7.1-.../lib:$LD_LIBRARY_PATH"  # FFmpeg
```

---

## 3. 使用方法

### 3.1 准备任务配置

为你的任务创建一个 JSON 配置文件，放在 `configs/` 目录下：

```json
{
    "activity": "补充可乐",
    "variation": "none",
    "step_headline": [
        "从黑色货架上拿一瓶可乐",
        "把可乐放到白色货架上"
    ]
}
```

字段说明：
- `activity`：任务名称，用于匹配先决依赖矩阵
- `variation`：任务变体（通常填 `"none"`）
- `step_headline`：步骤描述列表，**步骤顺序应与实际执行顺序一致**

### 3.2 生成先决依赖矩阵（推荐，但非必须）

先决依赖矩阵描述步骤之间的先后约束。用 LLM 一次性生成：

```bash
conda activate baglm
bash scripts/custom_run_llm_prereq.sh
```

这会为每个任务生成一个 `.pt` 文件。如果不生成，系统会用单位矩阵（无顺序约束），判断结果仍可用但精度稍低。

### 3.3 启动服务端

```bash
# InternVL 模型
conda activate baglm
bash scripts/online_start_server.sh

# RoboBrain 模型（推荐：更快、更适合机器人场景）
conda activate baglm-robobrain
bash scripts/online_start_server_robobrain.sh
```

不带先决依赖矩阵时（等同于 `--no-prereq`）：

```bash
bash scripts/online_start_server.sh  # 不传 --prereq_dir 即可
```

### 3.4 启动客户端

```bash
# 用视频文件测试
bash scripts/online_start_client.sh --video_path /path/to/video.mp4

# 用摄像头测试
bash scripts/online_start_client.sh --webcam

# 只测前 10 个段
bash scripts/online_start_client.sh --video_path /path/to/video.mp4 --max_segments 10

# 连接远程服务器
bash scripts/online_start_client.sh --host 192.168.1.100 --port 9999 --video_path video.mp4
```

### 3.5 查看日志和可视化

运行后自动生成日志：

```
logs/online/
├── Restock_Cola/                      # 服务端逐段日志
│   ├── 20260507_123727.jsonl          #   每行一个 segment 的结果
│   └── 20260507_123807.jsonl
├── session_20260507_123812_leftImg.mp4.json  # 客户端会话汇总
└── report_leftImg.html                       # HTML 可视化报告
```

生成可视化报告：

```bash
PYTHONPATH="src:t2v_metrics" python src/visualize_log.py \
    logs/online/session_20260507_123812_leftImg.mp4.json \
    -o logs/online/report.html
```

报告包含：统计卡片、步骤时间线、Belief 演化曲线、置信度/延迟图、逐段详情表。

### 3.6 机器人端集成（Python 示例）

```python
import socket, json, struct, numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("192.168.1.100", 9999))

# 采集 4 帧 (segment_duration=2 × sampling_fps=2)
frames = np.stack([get_camera_frame() for _ in range(4)])  # shape [4, 480, 640, 3]

# 发送
header = json.dumps({"n_frames": 4, "frame_h": 480, "frame_w": 640}).encode()
sock.sendall(struct.pack("!I", len(header)) + header)
sock.sendall(frames.tobytes())

# 接收
raw_len = sock.recv(4)
msg_len = struct.unpack("!I", raw_len)[0]
resp = json.loads(sock.recv(msg_len))

print(f"当前步骤: {resp['step']}, 置信度: {resp['confidence']:.3f}")
print(f"完整 belief: {resp['belief']}")
```

---

## 4. 参数配置

### 4.1 服务端参数 (`online_server.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址。`0.0.0.0` 表示所有网卡 |
| `--port` | `9999` | 监听端口 |
| `--task_config` | **必填** | 任务配置 JSON 文件路径 |
| `--model` | `internvl2.5-8b` | 模型名称。可选值见下方模型列表 |
| `--device` | `cuda` | 推理设备 |
| `--dep_matrix_path` | 无 | 先决依赖矩阵 `.pt` 文件的直接路径 |
| `--prereq_dir` | 无 | 先决依赖矩阵目录（自动查找 `<activity>_<variation>.pt`） |
| `--cache_dir` | `$HF_HOME` | HuggingFace 模型缓存目录 |
| `--segment_duration` | `2` | 每个时间段包含的秒数 |
| `--sampling_fps` | `2` | 每秒采样帧数 |
| `--visual_batch_size` | `1` | 视觉推理批大小 |
| `--text_batch_size` | `1` | 文本推理批大小 |
| `--vsg_prompt_file` | 自动 | VSG 任务的 prompt 文件路径 |
| `--prog_prompt_file` | 自动 | Progress 任务的 prompt 文件路径 |
| `--log_dir` | `logs/online` | 日志保存目录 |

`--dep_matrix_path` 和 `--prereq_dir` 二选一。都不传时使用单位矩阵。

### 4.2 客户端参数 (`online_client.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `localhost` | 服务端地址 |
| `--port` | `9999` | 服务端端口 |
| `--video_path` | 无 | 视频文件路径（与 `--webcam` 二选一） |
| `--webcam` | 无 | 使用摄像头模式 |
| `--webcam_id` | `0` | 摄像头设备号 |
| `--segment_duration` | `2` | 每段秒数（应与服务端一致） |
| `--sampling_fps` | `2` | 采样帧率（应与服务端一致） |
| `--max_segments` | `0` | 最大处理段数，`0` 表示全部 |
| `--log_dir` | `logs/online` | 日志保存目录 |

### 4.3 可用模型

| 模型名 | 参数量 | 显存 | 环境 | 特点 |
|--------|-------|------|------|------|
| `internvl2.5-8b` | 8B | ~17GB | `baglm` | 论文原始模型，通用 LMM |
| `internvl2.5-2b` | 2B | ~5GB | `baglm` | 轻量版，推理更快 |
| `robobrain2.5-4b` | 4B | ~9GB | `baglm-robobrain` | 专为机器人设计，**推荐** |
| `qwen2.5-vl-7b-instruct` | 7B | ~15GB | `baglm` | Qwen 视觉语言模型 |

### 4.4 关键参数组合建议

**快速响应（低延迟优先）**：

```bash
--model robobrain2.5-4b --segment_duration 1 --sampling_fps 1
# 每段 1 帧，推理 ~200ms
```

**高精度（准确率优先）**：

```bash
--model internvl2.5-8b --segment_duration 2 --sampling_fps 4
# 每段 8 帧，推理 ~2s
```

**默认平衡配置**：

```bash
--model robobrain2.5-4b --segment_duration 2 --sampling_fps 2
# 每段 4 帧，推理 ~400ms
```

---

## 5. 输出说明

### 5.1 服务端响应格式

每个 segment 推理后返回的 JSON：

```json
{
    "step": "Pick up one cola can from the black shelf",
    "step_idx": 0,
    "confidence": 0.750,
    "belief": {
        "Pick up one cola can from the black shelf": 0.75,
        "Put the cola can on the white shelf": 0.05,
        "<none>": 0.20
    },
    "segment_count": 1,
    "total_segments": 5,
    "timestamp": 1715051234.567,
    "latency_ms": 420.0
}
```

字段含义：

| 字段 | 含义 |
|------|------|
| `step` | 当前最可能的步骤名称（排除 none） |
| `step_idx` | 步骤索引（0-based） |
| `confidence` | 对当前步骤的 belief 概率（0-1），越高越确定 |
| `belief` | 所有步骤 + `<none>` 的完整概率分布，总和为 1 |
| `segment_count` | 本批次处理的段数 |
| `total_segments` | 本次会话累计处理的段数 |
| `latency_ms` | 本次推理耗时（毫秒），不含网络传输 |

### 5.2 如何解读置信度

- **> 0.5**：模型较确定当前步骤，可以直接使用判断结果
- **0.2 ~ 0.5**：有一定倾向但不明确，建议继续观察
- **< 0.2**：模型不确定，`<none>` 概率高，说明当前帧不像任何步骤

置信度会随着更多帧的积累（贝叶斯滤波累积效应）逐渐上升。

---

## 6. 文件结构

```
baglm/
├── configs/                          # 任务配置
│   ├── restock_cola.json
│   └── custom_task_template.json
├── scripts/
│   ├── online_start_server.sh        # InternVL 服务端启动脚本
│   ├── online_start_server_robobrain.sh  # RoboBrain 服务端启动脚本
│   └── online_start_client.sh        # 客户端启动脚本
├── src/
│   ├── online_server.py              # Socket 推理服务端
│   ├── online_client.py              # Socket 测试客户端（支持视频/摄像头）
│   ├── online_bayes.py               # 增量式贝叶斯滤波器
│   └── visualize_log.py             # 日志可视化（生成 HTML 报告）
├── logs/online/                      # 自动生成的日志和报告
│   ├── <任务名>/<时间戳>.jsonl       # 服务端逐段日志
│   ├── session_<时间戳>.json         # 客户端会话汇总
│   └── report_*.html                 # 可视化报告
└── prompts/
    ├── vsg/question_robot.txt        # VSG 机器人视角 prompt
    └── prog/question_robot.txt       # Progress 机器人视角 prompt
```
