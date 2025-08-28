# 阿拉伯语大语言模型评测维度与评判标准

本文件总结了近年来阿拉伯语大模型测评的 **主要维度** 及其 **评判标准**，按照“知识与理解、文化常识与价值观、推理与逻辑”分类。

---

## 1. 知识与理解（Knowledge & Understanding）

| 数据集 | 任务内容 | 评判标准 |
|--------|---------|-----------|
| **[ArabicMMLU (2024)](https://huggingface.co/datasets/MBZUAI/ArabicMMLU)** | 14,575 道多选题，覆盖 STEM、社科、人文等40项任务 | **准确率 (Accuracy)**：整体和分学科正确率 |
| **[AlGhafa (2023)](https://aclanthology.org/2023.arabicnlp-1.17/)** | 多任务选择题，覆盖常识、阅读理解、专业知识 | **准确率 / 宏平均F1** |
| **[RACE_Ar](https://aclanthology.org/2021.wanlp-1.22/)** | 长文阅读理解（改编自英文RACE） | **Accuracy、Exact Match (EM)** |
| **[ARCD](https://huggingface.co/datasets/hsseinmz/arcd)** | 阿语阅读理解（抽取式QA） | **EM + F1** |
| **[ArabicaQA (2024)](https://arxiv.org/abs/2403.01234)** | 开放域百科问答 | **EM / F1**（抽取式），**人工评估**（开放式） |

---

## 2. 文化常识与价值观（Culture & Value Alignment）

| 数据集 | 任务内容 | 评判标准 |
|--------|---------|-----------|
| **[ACVA (2023)](https://arxiv.org/abs/2305.00075)** | 文化价值观对齐测试 | **人工评分 / GPT-4评审** |
| **[ArabCulture (2025)](https://aclanthology.org/2025.acl-long.151/)** | 13国原创题，涵盖社会规范与文化常识 | **Accuracy**，可细分地区表现 |
| **[SaudiCulture (2025)](https://link.springer.com/article/10.1007/s00500-025-08951-2)** | 沙特全国与地区文化知识（单选/多选/开放问答） | **Accuracy**（单选），**部分正确率**（多选） |
| **[CamelEval (2024)](https://arxiv.org/abs/2405.10350)** | 开放式生成，含文化子集（方言、俗语） | **LLM-as-judge 自动评分**：指令遵循、事实正确性、文化贴合度 |
| **[Palm (2024)](https://arxiv.org/abs/2408.09876)** | 22国、20话题的文化指令任务 | **人工评估 / GPT-4评分**：文化敏感性、方言恰当性、指令完成度 |

---

## 3. 推理与逻辑（Reasoning & Logic）

| 数据集 | 任务内容 | 评判标准 |
|--------|---------|-----------|
| **[COPA_Ar](https://aclanthology.org/2021.wanlp-1.22/)** | 因果推理（选择可能原因/结果） | **Accuracy** |
| **Ar_Math (2022)** | 数学推理题（算术、代数、应用题） | **Accuracy**（最终答案正确率）、逐步推理一致性 |
| **[AraTable (2025)](https://arxiv.org/abs/2503.01234)** | 表格数据推理（算术、比较、多步推理） | **Accuracy**，分类型指标（简单 vs 多步推理） |
| **CamelEval（推理子集）** | 开放问答逻辑/事实性 | **自动化评分**：逻辑一致性、事实正确性 |

---

## 📌 总结

- **知识与理解**：选择题/阅读理解/QA → 指标多为 **Accuracy、F1、EM**  
- **文化常识与价值观**：文化与伦理 → 指标多为 **人工评估、自动打分、Accuracy**  
- **推理与逻辑**：数学/因果/表格推理 → 指标为 **Accuracy + 逻辑一致性**  

此维度划分和评判标准可作为研究和基准测试的参考。

