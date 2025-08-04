import os
import json

def clean_text(text):
    lines = text.split("\n")
    clean_lines = [line.strip() for line in lines if line.strip()]
    return clean_lines

def preprocess_and_save(raw_dir, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = []

    for fname in sorted(os.listdir(raw_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(raw_dir, fname), "r", encoding="utf-8") as f:
                text = f.read()
                segments = clean_text(text)
                for seg in segments:
                    # 可调整 instruction 和输出格式
                    results.append({
                        "instruction": "请用儿童语言讲一个小故事：",
                        "input": "",
                        "output": seg
                    })

    with open(save_path, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"保存 {len(results)} 条预处理文本到 {save_path}")

if __name__ == "__main__":
    preprocess_and_save(
        raw_dir="data/raw/arabic_ebooks_kids",
        save_path="data/processed/arabic_ebooks_kids/sft_format.jsonl"
    )
