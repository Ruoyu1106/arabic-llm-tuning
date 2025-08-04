# 📂 原始语料说明（data/raw）

本目录用于存放本项目中所有原始阿拉伯语语料库，主要用于儿童语言教育方向的语言模型微调任务。

---

## 1. 📘 `arabic_ebooks_kids/`

- 来源：Hugging Face 数据集 `mohres/The_Arabic_E-Book_Corpus`
- 子集筛选：已筛选出 `"children_stories"` 类别
- 格式：纯文本 `.txt` 文件，每个文件为一个故事
- 用途：可直接用于 SFT 微调，经过清洗后转为 `instruction/output` 格式

---

## 2. 📗 `arastories/`

- 来源：[UBC-NLP/arastories](https://github.com/UBC-NLP/arastories)
- 内容：包含阿拉伯语（MSA、方言）的结构化故事 CSV 文件，含 prompt、story、title 等字段
- 数据量：约 2,300 条故事样本
- 用途：适合构建多样化 prompt-based instruction 微调语料

---

## 3. 📙 `dares/`

- 来源：[DamithDR/arabic-readability-assessment](https://github.com/DamithDR/arabic-readability-assessment)
- 内容：来自沙特官方小学教材的简短语句，用于可读性建模任务
- 语言：以现代标准阿拉伯语（MSA）为主，适合简洁语义提取或文本分类任务
- 用途：可用于训练分类器/排序模型，辅助 LLM 推理合理性建模

---

## 📝 注意事项

- 所有数据保留原始格式，未进行结构转换
- 数据预处理脚本位于 `scripts/` 文件夹中
- 最终用于模型训练的格式（如 JSONL）存放于 `data/processed/`

