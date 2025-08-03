# Arabic LLM Landscape: Background Survey (2025)

This document summarizes the current state of Arabic large language models (LLMs), particularly those relevant for educational and child-friendly tasks. The findings will guide the fine-tuning and evaluation process of our Arabic-EduChatTuning project.

---

## 1. Model Overview & Accuracy

| Model                  | Params   | Open-source | Avg. Accuracy | Benchmark / Notes          |
|------------------------|----------|--------------|----------------|-----------------------------|
| **Qwen-Arabic (1.5B)** | ~1.5B    | ✅ Yes        | ~42.3%         | ArabicMMLU                  |
| **Qwen2.5-32B**        | ~32B     | ✅ Yes        | ~60.3%         | ILMAAM Benchmark            |
| **Qwen3 Series**       | 0.6–235B | ✅ Yes        | Very High      | Supports 128K context       |
| **Jais (13B/30B)**     | ~13–30B  | ✅ Yes        | ~62.3%         | Zero-shot QA                |
| **Juhaina**            | ~9B      | ✅ Yes        | High           | Best on CamelEval           |
| **ARBERT / MARBERT**   | <1B      | ✅ Yes        | 77.4 (ARLUE)   | Understanding tasks only    |

---

## 2. Benchmarks and Evaluation Frameworks

- **ArabicMMLU**: Multiple choice, 57 subjects, education-focused.
- **CamelEval**: Instruction-following, cultural/contextual relevance.
- **ARLUE**: Sentiment, NER, classification (used by MARBERT).
- **ILMAAM**: Designed for Middle East high school task types.

---

## 3. Summary Recommendations

- **Start point**: Use Qwen-Arabic 1.5B with child-directed corpus.
- **Evaluation**: Try ArabicMMLU for general, CamelEval for instruction following.
- **Scaling**: If resources allow, consider Qwen2.5 or Jais later.
- **Literature**: Focus on Jais, Qwen, and Arabic-specific NLP papers in early stages.

---

## 4. References

- Qwen-Arabic: https://github.com/prakash-aryan/qwen-arabic-project
- Qwen3 Blog: https://qwenlm.github.io/blog/qwen3/
- Jais LLM: https://en.wikipedia.org/wiki/Jais_(language_model)
- CamelEval: https://arxiv.org/abs/2409.12623
- ARBERT: https://arxiv.org/abs/2101.01785
