import os
import json
import pandas as pd

def clean_text(text):
    return (
        str(text)
        .replace("\\n", " ")  # 替换双转义换行
        .replace("\n", " ")   # 替换实际换行
        .replace("\r", "")    # 删除回车符
        .strip()
    )

def convert_csv_to_jsonl(csv_path, jsonl_path):
    df = pd.read_csv(csv_path)

    records = []
    for _, row in df.iterrows():
        instruction = clean_text(row.get("Prompt", ""))
        output = clean_text(row.get("Story", ""))
        if instruction and output:
            records.append({
                "instruction": instruction,
                "input": "",
                "output": output
            })

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 保存 {len(records)} 条样本到：{jsonl_path}")

if __name__ == "__main__":
    base_input = "data/raw/arastories/data"
    base_output = "data/processed/arastories"

    convert_csv_to_jsonl(
        os.path.join(base_input, "Formatted-Egyptian-prompts-stories-for-fine-tuning.csv"),
        os.path.join(base_output, "egyptian.jsonl")
    )
    convert_csv_to_jsonl(
        os.path.join(base_input, "Formatted-Moroccan-prompts-stories-for-fine-tuning.csv"),
        os.path.join(base_output, "moroccan.jsonl")
    )
    convert_csv_to_jsonl(
        os.path.join(base_input, "Formatted-MSA-prompts-stories-for-fine-tuning.csv"),
        os.path.join(base_output, "msa.jsonl")
    )
