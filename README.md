# Arabic LLM Tuning

This project explores how to fine-tune existing large language models (LLMs) for Arabic-language tasks, especially within the education domain.

## Objectives

- Investigate existing Arabic LLMs (e.g., AraBERT, CAMeLBERT, QARiB)
- Fine-tune multilingual models (e.g., ChatGLM/Qwen) on Arabic corpora
- Evaluate model performance and limitations on Arabic tasks
- Build a useful pipeline for Arabic LLM customization

## Project Structure

```
arabic-llm-tuning/
├── data/         # Datasets (raw/cleaned)
├── notebooks/    # Jupyter notebooks for exploration
├── models/       # Model configs and checkpoints
├── scripts/      # Fine-tuning and training scripts
├── papers/       # Literature reviews and notes
└── resources/    # Arabic corpora and useful links
```

## Setup

This project uses Python 3.10+ and Hugging Face Transformers.

```bash
# Clone the repo and install dependencies
git clone https://github.com/Ruoyu1106/arabic-llm-tuning.git
cd arabic-llm-tuning
pip install -r requirements.txt
```

## Recommended Libraries

- transformers
- datasets
- accelerate
- peft (for LoRA fine-tuning)
- scikit-learn
- matplotlib / wandb for visualization

## Next Steps

- [ ] Collect Arabic educational datasets
- [ ] Review and compare Arabic LLMs
- [ ] Start fine-tuning experiments
