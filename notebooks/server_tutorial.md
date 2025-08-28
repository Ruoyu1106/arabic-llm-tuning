# 服务器上部署阿拉伯语大模型与评测全流程教程
（最后更新：2025-08-28）

本教程手把手带你在一台云服务器（如阿里云、AWS、GCP）上**部署开源阿拉伯语/多语 LLM**（如 Qwen、Jais、Juhaina、Llama3 系列等），并通过 **lm-evaluation-harness + vLLM** 进行批量评测，自动产出 **JSON/CSV** 结果表，还包含**常见故障排查**。

---

## 0. 硬件与系统建议
- **GPU**：
  - 7B/8B 模型：≥ 24GB 显存（如 L4/A10/A4000/A5000）；
  - 13B/14B：≥ 40GB（A100 40G 更稳）；
  - 显存不足可用 **8-bit/4-bit 量化** 或 **张量并行**。
- **CPU/内存**：8 vCPU / 32GB RAM 起步更顺畅。
- **磁盘**：≥ 200GB（模型权重 + 数据缓存）。
- **系统**：Ubuntu 22.04 LTS；NVIDIA 驱动/CUDA 对应 PyTorch 版本。

> 若不确定 CUDA 版本与 PyTorch 兼容关系，可先安装“带 CUDA 的官方 PyTorch 轮子”，再装 vLLM/Transformers。

---

## 1. 基础环境安装

### 1.1 系统依赖
```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential tmux htop nvtop screen wget curl python3-venv
git lfs install
```

### 1.2 Conda 或 venv
**任选其一**：
```bash
# A) Miniconda（推荐）
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda init
# 新开一个 shell 后继续：
conda create -n arabic-llm-eval python=3.10 -y
conda activate arabic-llm-eval

# B) Python venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 1.3 Python 依赖（评测 & 推理）
```bash
pip install "torch"  # 若失败，参考 pytorch.org 选择与你 CUDA 匹配的命令
pip install -U transformers accelerate datasets evaluate peft bitsandbytes
pip install -U vllm lm-eval pandas numpy matplotlib git-lfs
```

> 验证：
```bash
python - <<'PY'
import torch, vllm, transformers
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("vLLM:", vllm.__version__)
print("Transformers:", transformers.__version__)
PY
```

---

## 2. 模型准备（Qwen / Jais / Juhaina / Llama 等）

### 2.1 Hugging Face 登录（如需）
部分模型需登录/接受 license：
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

### 2.2 推荐首批模型（可按需增删）
- `Qwen/Qwen2.5-7B-Instruct`（多语较强）
- `inceptionai/jais-13b-chat`（阿英双语聊天）
- `meta-llama/Llama-3.1-8B-Instruct`（通用对照）
- `elmresearchcenter/juhaina`（阿拉伯文化对齐取向）
- （理解类对照）`UBC-NLP/MARBERT` / `aubmindlab/bert-base-arabertv02`（BERT 家族，非生成）

> 生成式评测建议优先 Instruct/Chat 版本；分类/理解可补充 BERT 系列。

---

## 3. 评测任务配置

我们使用 `configs/tasks.yaml` 与 `configs/models.yaml` 来管理要跑的**任务集**与**模型参数**。

### 3.1 任务集（可编辑）
见 `configs/tasks.yaml`。你可通过 `lm_eval --list_tasks | grep -i arabic` 查找可用任务名。

### 3.2 模型参数（可编辑）
见 `configs/models.yaml`。`model_args` 会传给 vLLM（如 `gpu_memory_utilization=0.9,dtype=float16`）。

---

## 4. 一键运行评测

### 4.1 运行脚本（最常用）
```bash
bash scripts/run_lm_eval.sh qwen2.5-7b-instruct arabic-core
```
- 第1个参数：`models.yaml` 的 key（如 `qwen2.5-7b-instruct`）。  
- 第2个参数：`tasks.yaml` 的任务集 key（如 `arabic-core`）。  
- 结果会保存在 `results/<MODEL_KEY>/` 下，自动导出 **JSON + CSV**。

### 4.2 常见可选参数
在 `run_lm_eval.sh` 第3个参数 `EXTRA_ARGS` 传递：
- `--limit 200`：只取前 200 条做 **smoke test**；
- `--batch_size auto`：默认；显存紧张可设小一些；
- `--apply_chat_template`：给 chat 模型加对话模板；
- `--num_fewshot 0/5`：few-shot 数量。

示例：
```bash
bash scripts/run_lm_eval.sh jais-13b-chat arabic-culture "--limit 300 --apply_chat_template --num_fewshot 0"
```

---

## 5. 使用 vLLM 部署推理服务（可选）

如需提供 OpenAI 兼容 API，运行：
```bash
bash scripts/run_vllm_server.sh qwen2.5-7b-instruct 8000
# 然后用 OpenAI 格式 SDK 调用 http://<server_ip>:8000/v1/chat/completions
```

---

## 6. 结果组织与可视化

- 每次评测输出一个带时间戳的 JSON：`results/<model>/<model>_<taskset>_<UTC>.json`
- 脚本会自动导出 `*.csv`，包含每个任务的关键指标（如 Accuracy/F1/EM 等）。
- 你可以把多次评测的 CSV 合并做折线/柱状图（示例见 `scripts/plot_results.py`）。

---

## 7. 故障排查（FAQ）

- **CUDA OOM**：
  - 降低 `--batch_size`；
  - 在 `models.yaml` 中把 `dtype` 设为 `float16` 或 `bfloat16`；
  - 尝试 8-bit/4-bit（`load_in_8bit` / `load_in_4bit`，注意 lm-eval 与 vLLM 的支持差异）。
- **Transformers 版本不兼容**：
  - 升级 `transformers` 与 `accelerate`；
  - 若模型需要 `trust_remote_code=True`，请在 `models.yaml` 中加入 `trust_remote_code=True`（对于 vLLM，可用 `--trust-remote-code` 或在 `model_args`里加）。
- **任务别名不匹配**：
  - 先 `lm_eval --list_tasks | grep -i arabic` 确认名称；
  - 有些数据需下载/缓存较久，请耐心等待或切换镜像源。
- **速度慢**：
  - 确认在使用 GPU (`nvidia-smi` 查看占用)；
  - 调整 `gpu_memory_utilization`（如 0.90 → 0.95）；
  - 开启 tensor parallel （多卡时，`tensor_parallel_size=n`）。

---

## 8. 最小可复现 Demo

### 8.1 仅跑 50 条（smoke test）
```bash
bash scripts/run_lm_eval.sh llama-3.1-8b-instruct arabic-core "--limit 50 --num_fewshot 0"
```

### 8.2 评测文化面板
```bash
bash scripts/run_lm_eval.sh qwen2.5-7b-instruct arabic-culture "--apply_chat_template --limit 200"
```

### 8.3 表格推理（如支持）
```bash
bash scripts/run_lm_eval.sh qwen2.5-7b-instruct arabic-reasoning "--limit 200"
```

---

## 9. 目录结构建议
```
arabic-llm-tuning/
├─ README.md
├─ docs/
│  ├─ evaluation_dimensions.md
│  └─ server_tutorial.md        # ← 本文档
├─ configs/
│  ├─ tasks.yaml
│  └─ models.yaml
├─ scripts/
│  ├─ setup_server.sh
│  ├─ install_env.sh
│  ├─ run_lm_eval.sh
│  ├─ run_vllm_server.sh
│  └─ plot_results.py
└─ results/
   └─ ...
```
把本文件夹全部拷到你的仓库即可。
