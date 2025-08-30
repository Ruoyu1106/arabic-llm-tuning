# Arabic LLM Tuning

本项目旨在研究与评估 **阿拉伯语大型语言模型（LLMs）**，重点关注其在 **教育领域** 的应用。项目目标包括：基准评测、差距分析以及针对性微调与部署管线的构建。

---

## 📌 研究目标
- 调研与对比：现有阿拉伯语与多语言 LLM（如 AraBERT、CAMeLBERT、QARiB、Qwen、ChatGLM 等）。
- 系统评测：在标准化阿拉伯语基准（如 AraMMLU、CIDAR、方言任务等）上进行多任务测试（分类、问答、生成、推理）。
- 构建评测管线：打通数据预处理、推理评测、指标计算与可视化，保证可复现性。
- 识别不足：定位教育场景下的薄弱点（如指令跟随、事实一致性、方言鲁棒性）。
- 探索改进：采用 LoRA/QLoRA、指令微调等方法优化模型表现，并记录消融实验。

---

## 📂 项目结构
```text
arabic-llm-tuning/
├── data/         # 数据集（raw/cleaned）
├── notebooks/    # 探索与实验（Jupyter）
├── models/       # 模型配置与检查点
├── scripts/      # 训练 / 评测 / 预处理脚本
├── papers/       # 文献阅读与笔记
└── resources/    # 语料、基准与工具链接
```

---

## 🔄 当前进展
- ✅ 文献调研与仓库/环境初始化
- ✅ 评测管线初版（`transformers` + `lm-eval-harness`/自定义脚本）
- ✅ 多模型零样本基线测试（Arabic 专用 vs 多语言）
- 🔄 模型对比与误差分析（教育任务）
- 🔜 收集与清洗 **阿拉伯语教育类数据集**
- 🔜 启动微调实验（LoRA/QLoRA、指令微调）

---

## ⚙️ 环境设置
本项目基于 **Python 3.10+** 与 Hugging Face 生态。

```bash
# 克隆仓库并安装依赖
git clone https://github.com/<your-username>/arabic-llm-tuning.git
cd arabic-llm-tuning
pip install -r requirements.txt
```

可选：使用 `accelerate` 进行分布式/混合精度推理与训练。

---

## 📚 推荐依赖
- transformers
- datasets
- accelerate
- peft  （LoRA/QLoRA）
- scikit-learn
- matplotlib / wandb

> 可在 `requirements.txt` 中统一管理版本。

---

## 🗺️ 路线图（Roadmap）
- [ ] 完善多任务评测管线与日志记录（可视化/对比图表）。
- [ ] 收集与标准化教育任务数据集（含数据卡 Data Card）。
- [ ] 完成 Arabic 专用与多语言模型的分任务对比评测。
- [ ] 进行 LoRA/QLoRA / 指令微调与消融实验。
- [ ] 发布基准结果、误差分析与复现实验脚本。

---

## 📊 结果与发现（预留）
> 在此填入你的实验结果，建议保留 **表格 + 文字要点**。

### 汇总表（示例）
| 模型 | 数据集 | 任务 | 准确率 | F1 | Rouge-L | 备注 |
|---|---|---|---:|---:|---:|---|
| AraBERT | AraMMLU | 分类 | xx.x | xx.x | — | baseline |
| Qwen-7B | AraMMLU | 分类 | xx.x | xx.x | — | 多语言对比 |
| CAMeLBERT | CIDAR | QA | xx.x | xx.x | — | 教育任务 |
| Qwen-7B-QLoRA | CIDAR | QA | xx.x | xx.x | — | 指令微调 |
| QARiB | Dialect | 生成 | — | — | xx.x | 方言鲁棒性 |

**要点摘录（示例）**
- 在 **教育类 QA** 上，指令微调带来 **+Δ** 的提升，主要体现在……
- 多语言模型在 **零样本** 上对 **方言子集** 更稳健/不稳健，原因可能与……
- 错误类型主要集中在 **实体混淆 / 语义脱靶 / 指令误解**，建议通过…… 进行缓解。

---

## ▶️ 快速开始（示例）
### 评测（零样本）
```bash
python scripts/eval_zero_shot.py \
  --model qwen/Qwen2-7B-Instruct \
  --task arabicmmlu \
  --data_path data/arabicmmlu \
  --output runs/qwen2-7b_inst_zero_shot.json
```

### 微调（LoRA/QLoRA，示例）
```bash
python scripts/finetune_lora.py \
  --model qwen/Qwen2-7B-Instruct \
  --train_file data/edu/train.jsonl \
  --eval_file data/edu/dev.jsonl \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --output_dir models/qwen2-7b-edu-lora
```


