# BaGLM 复现文档

> 论文：Training-free Online Video Step Grounding (NeurIPS 2025)
> 仓库：https://github.com/lucazanella/baglm

---

## 一、整体框架

### 1.1 任务定义

给定一段视频和一个任务（如"补货可乐"）及其步骤列表（如"拿起第一罐可乐"、"放到货架上"等），
BaGLM 在线逐帧预测当前正在执行哪个步骤。核心特点：**不需要训练**，利用现有大模型的零样本能力 + 贝叶斯滤波融合时序信息。

### 1.2 整体 Pipeline

```text
输入：视频 + 任务名 + 步骤列表
       │
       ├──→ [模块A] LMM VSG 预测     → results/vsg/{model}/{video_uid}.pt
       ├──→ [模块B] LMM Progress 预测 → results/prog/{model}/{video_uid}.pt
       ├──→ [模块C] LLM Prereq 预测   → results/prereq/{model}/{activity}_{variation}.pt
       │         (A/B/C 彼此独立，可并行)
       └──→ [模块D] 贝叶斯滤波        → 融合上述三者，输出最终预测
                └──→ [模块E] 可视化导出 → CSV / JSON / HTML 时间轴
```

### 1.3 各模块详解

#### 模块A：VSG（Video Step Grounding）预测

| 项目 | 内容 |
|------|------|
| **脚本** | `src/custom_eval.py` + `--prompt_type vsg` |
| **输入** | 视频帧 + 步骤列表 + prompt 模板 (`prompts/vsg/question.txt`) |
| **输出** | `results/vsg/{model}/{video_uid}.pt`，形状 `[T, S+1]` |
| **含义** | T=时间步数（按秒），S=步骤数，+1 是 "None of the above"。每行是一个时间步上各步骤的 softmax 概率 |
| **原理** | 对每个时间段，给 LMM 看视频帧 + 多选题，让模型选"当前在执行哪个步骤" |

prompt 模板：
```text
You are watching a video segment of someone attempting to {goal}.
What is the main action being performed in this exact moment?
Options:
A. Pick up the first can...
B. Put the first can...
...
Z. None of the above.
Answer with only the letter label.
```

#### 模块B：Progress（进度）预测

| 项目 | 内容 |
|------|------|
| **脚本** | `src/custom_eval.py` + `--prompt_type prog` |
| **输入** | 视频帧 + 每个步骤单独询问 + prompt 模板 (`prompts/prog/question.txt`) |
| **输出** | `results/prog/{model}/{video_uid}.pt`，形状 `[T, S, 10]` |
| **含义** | 10 个进度档位（0=未开始, 9=即将完成），每步每时刻一个分布 |
| **原理** | 对每个时间段和每个步骤，让 LMM 评估该步骤的执行进度 0-9 |

#### 模块C：Prereq（依赖关系）预测

| 项目 | 内容 |
|------|------|
| **脚本** | `src/custom_qa.py` |
| **输入** | 任务名 + 步骤列表（纯文本，不需要视频） |
| **输出** | `results/prereq/{model}/{activity}_{variation}.pt`，形状 `[S, S]` |
| **含义** | `dep[i,j]` 表示 step_j 是 step_i 的前置条件的概率 |
| **原理** | 用 LLM 对每对步骤问"step_b 是否是 step_a 的前置条件"，取 Yes 的概率 |

#### 模块D：贝叶斯滤波（BaGLM 核心）

| 项目 | 内容 |
|------|------|
| **脚本** | `src/bayes_filter.py` |
| **输入** | VSG `[T,S+1]` + Prog `[T,S,10]` + Prereq `[S,S]` |
| **输出** | `beliefs [T, S+1]`（信念轨迹） → 终端打印 Recall@1 |
| **原理** | 见下方算法说明 |

```text
初始化: belief = 均匀分布 [1/S, ..., 1/S]

对每个时间步 t:
  1. 读取 VSG 分数 → vsg_scores_t
  2. 从 Prog 计算期望进度 → monotonic_progress（单调递增）
  3. 用依赖矩阵 + 进度计算：
     - readiness：前置条件完成程度（前置步骤进度越高→越就绪）
     - validity：步骤是否仍有效（后续步骤已完成→当前步骤可能过时）
  4. 动态转移矩阵 = 静态转移矩阵 × readiness × validity
  5. 预测: prior = belief @ transition        （贝叶斯预测步）
  6. 更新: posterior = normalize(prior × vsg)  （贝叶斯更新步）
  7. belief = posterior
```

#### 模块E：可视化导出

| 项目 | 内容 |
|------|------|
| **脚本** | `src/custom_export_predictions.py` |
| **输入** | VSG + Prog + Prereq 的 `.pt` 文件 |
| **输出** | `{video_uid}_timeline.html`（可视化时间轴）、`{video_uid}_timeline.csv`、`{video_uid}_predictions.json` |

### 1.4 模型封装 (`t2v_metrics/`)

| 子模块 | 支持的模型 | 用途 |
|--------|-----------|------|
| `vqascore_models/internvl_model.py` | InternVL2.5-8B, InternVL3-8B | 多模态 VQA（视频+文本→分数），用于 VSG 和 Progress |
| `vqascore_models/qwen2vl_model.py` | Qwen2.5-VL-7B | 同上 |
| `vqascore_models/llavaov_model.py` | LLaVA-OneVision-7B | 同上 |
| `qascore_models/gpt4_model.py` | GPT-4.1-mini（API） | 纯文本 QA，用于 Prereq |
| `qascore_models/llama33_model.py` | LLaMA-3.3-70B-Instruct | 本地纯文本 QA，用于 Prereq |

### 1.5 目录结构

```text
baglm/
├── src/
│   ├── custom_dataset.py        # 自定义数据 Dataset（视频平铺在一个目录下）
│   ├── custom_eval.py           # LMM 推理（VSG / Progress）
│   ├── custom_qa.py             # LLM 依赖矩阵生成
│   ├── custom_recall.py         # Recall@1 计算
│   ├── custom_export_predictions.py  # 可视化导出
│   ├── generate_video_annots.py # 自动生成 video_annots.json 模板
│   ├── bayes_filter.py          # BaGLM 贝叶斯滤波（已加 custom 支持）
│   ├── dataset.py               # 原版 Dataset（HTStep/CrossTask/COIN/Ego4D）
│   ├── htstep_eval.py / crosstask_eval.py / ...  # 原版 eval
│   └── utils/
│       ├── video_utils.py       # 视频解码（torchcodec）
│       └── text_utils.py        # 文本处理
├── scripts/
│   ├── custom_run_lmm_vsg.sh    # Step 3
│   ├── custom_run_lmm_prog.sh   # Step 4
│   ├── custom_run_llm_prereq.sh # Step 5
│   ├── custom_run_bayes_filter.sh  # Step 6
│   └── custom_export_predictions.sh  # Step 7
├── prompts/
│   ├── vsg/question.txt         # VSG 多选题模板
│   ├── prog/question.txt        # 进度评分模板
│   └── prereq/system.txt + question.txt  # 依赖关系问答模板
├── t2v_metrics/                 # 模型封装库
├── datasets/                    # 预处理后的 video_annots.json
└── requirements.txt
```

---

## 二、自定义数据的完整处理流程

### 2.0 环境准备

```text
服务器环境：
  - GPU: NVIDIA A100-SXM4-80GB
  - CUDA: 12.6
  - Python: 3.13 (conda env: baglm)
  - Conda 路径: /mnt/public1/dais/miniconda3/envs/baglm
  - 代码路径: /home/dais/workspace/baglm
  - 数据路径: /mnt/public1/dais/baglm_data/
  - HF 模型缓存: /mnt/public1/dais/hf_cache
```

环境已安装的关键依赖：
- PyTorch 2.6.0 + CUDA 12.6
- torchcodec 0.2.1（视频解码，需 nvidia-npp-cu12）
- flash-attn 2.7.4.post1
- transformers 4.49.0
- InternVL2.5-8B 模型（HuggingFace）

conda activate 时自动设置 `LD_LIBRARY_PATH`（nvidia-npp 库路径），无需手动配置。

### 2.1 Step 1：准备视频文件

将视频放到一个扁平目录下（不需要子目录）：

```text
/mnt/public1/dais/baglm_data/restock_cola/videos/
└── leftImg.mp4
```

### 2.2 Step 2：生成 video_annots.json

```bash
conda activate baglm
cd /home/dais/workspace/baglm

python src/generate_video_annots.py \
    --video_dir /mnt/public1/dais/baglm_data/restock_cola/videos \
    --output /mnt/public1/dais/baglm_data/restock_cola/video_annots.json
```

脚本自动用 ffprobe 提取 `num_frames`、`fps`、`duration`，生成 JSON 模板。
然后手动编辑，填入 `activity` 和 `step_headline`：

```json
[
  {
    "video_uid": "leftImg",
    "video_ext": ".mp4",
    "video_num_frames": 1437,
    "video_fps": 20.0,
    "video_duration": 71.85,
    "activity": "Restock Cola",
    "variation": "none",
    "step_headline": [
      "Pick up the first can of cola from the shelf",
      "Put the first can of cola on the shelf",
      "Pick up the second can of cola from the shelf",
      "Put the second can of cola on the shelf",
      "Pick up the third can of cola from the shelf",
      "Put the third can of cola on the shelf"
    ],
    "clips": [{ "annotations": [{ "language_queries": [] }] }]
  }
]
```

字段说明：
- `video_uid`：视频文件名（不含扩展名），必须和文件名一致
- `activity`：任务描述
- `step_headline`：所有候选步骤列表
- `clips.language_queries`：GT 标注（仅计算 Recall@1 时需要，否则留空 `[]`）

### 2.3 Step 3：生成 VSG 分数（需要 GPU）

```bash
bash scripts/custom_run_lmm_vsg.sh
```

- 加载 InternVL2.5-8B 模型
- 对视频每 2 秒一个段，给 LMM 多选题
- 输出 `results/vsg/internvl2.5-8b/{video_uid}.pt`

### 2.4 Step 4：生成 Progress 分数（需要 GPU）

```bash
bash scripts/custom_run_lmm_prog.sh
```

- 复用同一模型
- 对每个段的每个步骤问进度 0-9
- 输出 `results/prog/internvl2.5-8b/{video_uid}.pt`

> Step 3 和 Step 4 都加载 InternVL，单卡上建议顺序跑。

### 2.5 Step 5：生成依赖矩阵（需要 API 或本地 LLM）

```bash
bash scripts/custom_run_llm_prereq.sh
```

- 默认使用 GPT-4.1-mini（通过 OpenAI 兼容 API）
- API Base URL 配置在脚本中：`OPENAI_BASE_URL="https://api.vectorengine.ai/v1"`
- API Key 放在：`/home/dais/workspace/baglm/OPENAI_KEY.txt`
- 纯文本，不需要视频，不需要 GPU
- 输出 `results/prereq/gpt-4.1-mini/{activity}_{variation}.pt`

> Step 5 和 Step 3/4 完全独立，可以并行或先跑。

### 2.6 Step 6：运行贝叶斯滤波

```bash
bash scripts/custom_run_bayes_filter.sh
```

- 仅 CPU 计算，读取 Step 3/4/5 的 `.pt` 文件
- 终端输出 `Mean Recall@1`（需要 GT 标注才有意义）

### 2.7 Step 7：导出可视化结果

```bash
bash scripts/custom_export_predictions.sh
```

输出到 `results/visualization/`：

| 文件 | 用途 |
|------|------|
| `{video_uid}_timeline.html` | 浏览器打开，可视化时间轴 |
| `{video_uid}_timeline.csv` | 表格分析，每行一个时间段 |
| `{video_uid}_predictions.json` | 完整数据（含 progress 向量） |
| `all_videos_timeline.csv` | 所有视频汇总 |

### 2.8 运行顺序总结

```text
Step 1: 准备视频               → videos/ 目录
Step 2: 生成 video_annots.json → 手动填 activity + step_headline
   ↓
Step 3: VSG 分数  ─┐
Step 4: Prog 分数  ├─ 彼此独立，3/4 需 GPU，5 需 API
Step 5: Prereq    ─┘
   ↓
Step 6: Bayes Filter           → 终端打印 Recall@1
Step 7: 导出可视化              → HTML / CSV / JSON
```

### 2.9 更换数据集

只需修改每个 `.sh` 脚本中的 `DATA_DIR`：

```bash
DATA_DIR="/mnt/public1/dais/baglm_data/你的新数据目录"
```

其他路径（VIDEO_DIR、VIDEO_ANNOTS_FILE、RESULT_DIR）都基于 DATA_DIR 自动拼接。

---

## 三、用预计算数据复现论文结果

如果只想验证论文指标，不需要 GPU 和视频：

1. 从 Google Drive 下载预计算 zip（含 vsg/prog/prereq 的 `.pt`）
2. 解压到 `/mnt/public1/dais/baglm_data/{dataset}/results/`
3. 运行贝叶斯滤波：

```bash
cd /home/dais/workspace/baglm
source /mnt/public1/dais/miniconda3/bin/activate baglm
export PYTHONPATH="$PWD/src:$PWD/t2v_metrics:$PYTHONPATH"

python src/bayes_filter.py \
    --dataset "htstep" \
    --model "internvl2.5-8b" \
    --video_annots_file "datasets/htstep/htstep_video_annots.json" \
    --lmm_vsg_dir "/mnt/public1/dais/baglm_data/htstep/results/vsg/" \
    --lmm_prog_dir "/mnt/public1/dais/baglm_data/htstep/results/prog/" \
    --llm_prereq_dir "/mnt/public1/dais/baglm_data/htstep/results/prereq/llama-3.3-70b-instruct/"
```

已验证结果：

```text
HT-Step: Mean Recall@1 = 57.4%  (600/600 videos)
```

| 数据集 | Google Drive ID |
|--------|----------------|
| HT-Step | `1n5R7BV3eGwAw8YTT7KItiN3oY-pU_-oF` |
| CrossTask | `1bGkrEQd79SP0-Q1ekRKfShhz7MqojCg4` |
| Ego4D Goal-Step | `1LyNF025uHGqoVmn9TyvLyPxgqD2XHd3j` |
| COIN | `1jDo0at9gq-FlkhZkrEMbNH_kobXsPiuh` |
