# JAIS-6.7B Chat Benchmark Results on Arabic-MMLU

## 模型介绍
- **模型**: [JAIS-6.7B Chat](https://huggingface.co/inceptionai/jais-family-6p7b-chat)  
- **规模**: 6.7B 参数  
- **类型**: Causal LM (Chat 优化版本，支持阿拉伯语与英语)  
- **评测框架**: EleutherAI `lm-eval-harness`  
- **运行环境**: NVIDIA A100 80GB, batch size = 1, fp16  

## 数据集介绍
- **Arabic-MMLU**: 翻译自 MMLU (Massive Multitask Language Understanding) 的阿拉伯语版  
- **领域**: 人文、社会科学、STEM、语言、常识等  
- **任务形式**: 多项选择 (multiple-choice QA)  
- **指标**: Accuracy (acc)

---

## 总体结果
| Group            | Accuracy |
|------------------|----------|
| 全部 (Overall)   | **56.2%** |

---

## 分组结果
| Group            | Accuracy |
|------------------|----------|
| Humanities       | 57.9%    |
| Language         | 54.3%    |
| Other            | 61.4%    |
| Social Sciences  | 55.3%    |
| STEM             | 52.0%    |

---

## 详细科目结果（部分展示）
| Subject                          | Accuracy |
|----------------------------------|----------|
| History (High School)            | 46.5%    |
| Islamic Studies                  | 75.7%    |
| Arabic Language (Primary School) | 62.3%    |
| General Knowledge                | 58.8%    |
| Economics (University)           | 49.0%    |
| Political Science (University)   | 72.4%    |
| Biology (High School)            | 42.2%    |
| Natural Science (Primary School) | 80.4%    |

---

## 总结
JAIS-6.7B Chat 在 **Arabic-MMLU** 上总体达到了 **56.2%** 的准确率，在 **宗教/伊斯兰学科 (75.7%)** 和 **小学自然科学 (80.4%)** 上表现较好，而在 **历史** 和 **生物学** 方面表现相对较弱。  
这说明模型在某些阿拉伯语教育领域具有较强的知识覆盖，但在科学学科上仍有提升空间。

