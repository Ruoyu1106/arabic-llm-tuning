# Jais: Arabic‑Centric Foundation and Instruction‑Tuned LLM (2023)

- **Authors**：Neha Sengupta, Sunil Kumar Sahu 及团队；由 Inception (G42)、MBZUAI 与 Cerebras 合作推出  
- **Year / Venue**：2023 / arXiv + 官方博客  
- **URL / DOI**：https://arxiv.org/abs/2308.16149 :contentReference[oaicite:1]{index=1}  

---

## 🧠 关键贡献与亮点

- 发布了两个开源模型：**Jais（基础版，13B 参数）**与**Jais‑chat（指令微调版）**，在多项阿拉伯语任务上领先目前所有开源模型。:contentReference[oaicite:2]{index=2}  
- 在 Arabic 任务上表现显著超过其它开源模型，其英文性能也匹配同规模英文模型，尽管训练中英文数据比例偏低。:contentReference[oaicite:3]{index=3}

---

## 🧪 方法与训练细节

- **架构**：基于 GPT‑3 样式 autoregressive 解码结构；使用 ALiBi 位置嵌入、SwiGLU 激活函数和最大更新参数化 (muP)。:contentReference[oaicite:4]{index=4}  
- **训练数据**：
  - 包括 Arab‑English‑Code 三语料混合，比例约 1:2:0.4；
  - 总量约 395B tokens，其中约 116B 是阿拉伯语（包括 18B 翻译生成的阿拉伯语）；英语 ~232B，代码 ~47B。:contentReference[oaicite:5]{index=5}  
- **基础训练**耗时约 21 天，使用 G42 + Cerebras 的 Condor Galaxy‑1 超级计算平台完成。:contentReference[oaicite:6]{index=6}

---

## 📊 实验结果与评估

- Jais‑chat 在 Arabic 多项任务中得分大幅领先所有其他开源及多语模型。:contentReference[oaicite:7]{index=7}  
- 在 Arabic 与 English 双语推理任务中，性能接近甚至超越同参数英文模型，表现出跨语言迁移效果良好。:contentReference[oaicite:8]{index=8}  
- 在 mãT-bench 风格评估中，Jais 家族生成质量持续提升（Arabic 回答质量显著增强）。:contentReference[oaicite:9]{index=9}

---

## ✍️ 实验观察与反思

- Jais 在阿拉伯语指令‑微调任务中表现强劲，非常适合作为后续教育语料微调的基础模型。
- 混合语料策略（阿拉伯语 + 英语 + 代码）值得借鉴，尤其对于理解孩子可能接触的双语/代码片段类型内容。
- 使用 ALiBi、SwiGLU、muP 等现代架构优化提升效果，这可为我们微调脚本的超参数设计提供参考。

---

## ✅ 与我们项目的关联

- **作为教育语调微调基础**：Jais-chat 已在 instruction-following 上表现优秀，适合进行儿童教育类微调。
- **架构参考**：其使用的 embedding 与激活策略适合资源有限环境，可供设计适配。
- **训练数据启示**：可考虑在儿童教育语料中也采用 bilingual 或 code-mixed 样本，以提高模型泛化能力。

---

## 🔗 参考链接

- Sengupta 等人论文 “Jais and Jais-chat…”（arXiv）:contentReference[oaicite:10]{index=10}  
- G42/Cerebras 联合发布博客文章 “Jais: a New Pinnacle…” :contentReference[oaicite:11]{index=11}  
