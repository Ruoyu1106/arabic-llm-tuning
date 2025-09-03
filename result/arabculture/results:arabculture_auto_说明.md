
# ArabCulture 测评（5 个阿拉伯语大模型）——方法与结果说明

本文记录我们在 **MBZUAI/ArabCulture** 数据集上的评测方法、每一步做了什么，以及 5 个模型的最终准确率。评测脚本为自动校准版（`eval_arabculture_auto.py`），统一输出到规范化标签 **("نعم", "لا")**，并在必要时进行**极性翻转校正**（`flipped`）。

---

## 数据集简介

- **数据集**：Hugging Face 上的 `MBZUAI/ArabCulture`  
- **任务类型**：二分类（与阿拉伯文化/常识相关的判断题）  
- **国家配置**：13 个（Algeria, Egypt, Jordan, KSA, Lebanon, Libya, Morocco, Palestine, Sudan, Syria, Tunisia, UAE, Yemen）  
- **本次评测样本**：合并 13 个国家的 **test split**，总计 **3482** 个样本  

> 说明：每条样本本质上是一个“是/否”的判断。不同来源可能使用阿拉伯语的不同表达（如 “صح/خطأ”、“نعم/لا”，以及带空格/不带空格的变体），因此需要做统一与校准。

---

## 评测脚本做了什么（一步步解释）

1. **加载数据**  
   - 逐个加载 13 个国家的 `test` 分片，并合并为一个总的评测集（共 3482 条）。

2. **答案规范化（Canonicalization）**  
   - 我们把所有候选答案统一映射到规范化标签对：**("نعم", "لا")**。  
   - 针对数据与模型可能出现的不同表达（如 “صح/خطأ”、前后带空格的 “ نعم/لا ” 等），脚本在内部做了字符串清洗与同义映射。

3. **自动校准（Calibration, `calib_n=200`）**  
   - 在正式评测前，**随机抽取约 200 个样本** 做一个快速校准，测试多种**标签书写方式**与**正负极性**（是否需要把“是/否”对调）。  
   - 核心思想：先看看模型更“习惯”使用哪一套答案形式。如果发现模型输出与我们的规范化方向**极性相反**，就记为 **`flipped=True`**，并在后续**自动纠正**。  
   - 这一步可以有效避免因为“标签表达不一致”导致的整体准确率被严重拉低（例如之前你看到的 ~30% 现象）。

4. **全量评测（3482 条）**  
   - 使用校准确定的最佳映射与极性设置，在**全量 test** 上计算准确率。  
   - 即使模型原始更偏好 “صح/خطأ”，最终也会被规范为 ("نعم","لا") 来计分。  
   - 如果 `flipped=True`，说明脚本在评分时做了**极性修正**，但对外统一显示 (“نعم” 代表 “真/是”，“لا” 代表 “否/假”)，便于横向对比。

5. **指标**  
   - **Accuracy**（正确率）。

6. **输出**  
   - 机器可读：`results/arabculture_auto.json`（包含每个模型的最终准确率与 flipped 标志）  
   - 本说明文档：`results/arabculture_auto_说明.md`

---

## 结果（本次运行）

- **总样本数**：3482  
- **校准样本数**：`calib_n = 200`  

| 模型 | 准确率（Accuracy） | 是否翻转 (flipped) |
|---|---:|:---:|
| inceptionai/jais-family-6p7b-chat | **0.9334** | 是 |
| tiiuae/Falcon3-7B-Instruct | **0.9888** | 否 |
| omarwaleed523/qwen3-8b-arabic-multitask | **0.9836** | 否 |
| ALLaM-AI/ALLaM-7B-Instruct-preview | **0.4779** | 否 |
| silma-ai/SILMA-9B-Instruct-v1.0 | **0.7926** | 否 |

> 读取 `results/arabculture_auto.json` 可得到同样的数值（并带 `flipped` 字段）。

---

## 为什么会“翻转”（flipped）？

- 有些模型对“真/假”的输出习惯与我们的记分规范**相反**（例如更倾向输出 “صح/خطأ” 并把“صح”理解为负类）。  
- 若不校正，准确率会被系统性拉低。  
- **自动校准**会在 200 条样本上比较多套写法与极性，选择命中更高的设定；若检测到极性不一致，则记录为 `flipped=True` 并在**整个 3482 样本**的评分时**自动反转**，从而得到公平、可对比的准确率。

---

## 复现实验

你本次使用的命令（示例）：

```bash
python scripts/eval_arabculture_auto.py \
  --models \
    inceptionai/jais-family-6p7b-chat \
    tiiuae/Falcon3-7B-Instruct \
    omarwaleed523/qwen3-8b-arabic-multitask \
    ALLaM-AI/ALLaM-7B-Instruct-preview \
    silma-ai/SILMA-9B-Instruct-v1.0 \
  --batch_size 64 \
  --dtype bfloat16 \
  --out results/arabculture_auto.json