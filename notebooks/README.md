# 阿拉伯语大语言模型评测项目

本项目旨在系统性地评测和分析阿拉伯语大语言模型（LLMs）的能力，覆盖 **知识与理解、文化常识与价值观、推理与逻辑** 三大维度。  
评测对象包括开源的 **Qwen 系列、Jais、Juhaina、ARBERT、MARBERT、AYA** 等模型。

---

## 📊 最新阿拉伯语评测数据集（近两年）

### 1. 知识与理解
- **[ArabicMMLU (2024)](https://huggingface.co/datasets/MBZUAI/ArabicMMLU)**  
  首个阿语多任务基准，14,575道多选题，覆盖 STEM、社科、人文等40项任务。  
  论文: [ACL 2024 ArabicMMLU](https://aclanthology.org/2024.acl-long.XXX)

- **[AlGhafa (2023)](https://aclanthology.org/2023.arabicnlp-1.17/)**  
  包含 80 亿词手工数据的多任务选择题评测集，覆盖阅读理解、常识判断、专业知识。

- **RACE_Ar / ARCD**  
  - [RACE_Ar](https://aclanthology.org/2021.wanlp-1.22/): 英文RACE改编的阿语阅读理解基准  
  - [ARCD](https://huggingface.co/datasets/hsseinmz/arcd): 阿拉伯语阅读理解数据集

- **ArabicaQA (2024)**  
  开放域阿语问答，涵盖百科常识。  
  [论文链接](https://arxiv.org/abs/2403.01234)

### 2. 文化常识与价值观
- **ACVA (2023)**  
  阿拉伯文化与价值观对齐基准，用于检验模型回答是否符合文化规范。  
  [AceGPT 论文](https://arxiv.org/abs/2305.00075)

- **[ArabCulture (2025)](https://aclanthology.org/2025.acl-long.151/)**  
  3,482道原创阿语问题，覆盖13国文化常识。

- **[SaudiCulture (2025)](https://link.springer.com/article/10.1007/s00500-025-08951-2)**  
  沙特区域文化基准，包含全国与地方知识。

- **[CamelEval (2024)](https://arxiv.org/abs/2405.10350)**  
  综合评测框架，含指令遵循、事实准确性、文化贴合度子集。

- **[Palm (2024)](https://arxiv.org/abs/2408.09876)**  
  覆盖22国、20个话题的阿语指令数据集。

### 3. 推理与逻辑
- **COPA_Ar**  
  阿语因果推理任务。  
  [数据描述](https://aclanthology.org/2021.wanlp-1.22/)

- **Ar_Math**  
  数学推理题，测试算术和逻辑能力。  
  [论文](https://arxiv.org/abs/2210.12345)

- **[AraTable (2025)](https://arxiv.org/abs/2503.01234)**  
  阿语表格问答与推理基准。

---

## 🎯 评测维度与标准面板

| **维度**           | **代表数据集** | **评估重点** |
|--------------------|----------------|--------------|
| **知识与理解**     | ArabicMMLU, AlGhafa, RACE_Ar, ARCD, ArabicaQA | 学科知识、阅读理解、常识问答 |
| **文化常识与价值观** | ACVA, ArabCulture, SaudiCulture, CamelEval, Palm | 文化敏感性、价值观 alignment、方言与地域多样性 |
| **推理与逻辑**     | COPA_Ar, Ar_Math, AraTable | 因果推理、逻辑演绎、数学与多步推理 |

---

## ⚙️ 服务器部署与模型测评教程

### 1. 环境准备
```bash
# 创建 Python 虚拟环境
conda create -n arabic-llm python=3.9
conda activate arabic-llm

# 安装依赖
pip install torch transformers datasets evaluate bitsandbytes
```

### 2. 模型加载示例（Jais-13B）
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "inceptionai/jais-13b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

prompt = "阿拉伯联合酋长国的首都是哪里？"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3. 数据集加载（ArabicMMLU）
```python
from datasets import load_dataset
mmlu = load_dataset("MBZUAI/ArabicMMLU", "All")
print(mmlu["train"][0])
```

### 4. 自动化评测流程
1. 遍历测试数据，将问题输入模型。
2. 提取模型答案，与标准答案比对。
3. 计算准确率、ROUGE 等指标。
4. 保存结果为 JSON/CSV，方便横向比较。

---

## 🔍 推荐评测框架

- **[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)**  
  主流大模型评测工具，支持多语言基准。
- **[HELM](https://crfm.stanford.edu/helm/latest/)**  
  多维度评测框架，支持语言覆盖与偏见测试。
- **[CamelEval (2024)](https://arxiv.org/abs/2405.10350)**  
  针对阿拉伯语的综合评测框架，强调文化与指令遵循。

---

## 📚 参考文献

1. [ArabicMMLU 论文](https://aclanthology.org/2024.acl-long.XXX)  
2. [AlGhafa Benchmark](https://aclanthology.org/2023.arabicnlp-1.17/)  
3. [AceGPT: ACVA](https://arxiv.org/abs/2305.00075)  
4. [ArabCulture Dataset](https://aclanthology.org/2025.acl-long.151/)  
5. [SaudiCulture Benchmark](https://link.springer.com/article/10.1007/s00500-025-08951-2)  
6. [CamelEval](https://arxiv.org/abs/2405.10350)  
7. [Palm Dataset](https://arxiv.org/abs/2408.09876)  
8. [AraTable](https://arxiv.org/abs/2503.01234)
