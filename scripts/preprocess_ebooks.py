import os
import json
import re

def clean_text(text):
    """
    清理空行、不可见字符和控制符。
    """
    # 替换不可见字符为普通空格或删除
    text = text.replace('\u00A0', ' ')   # NBSP → 空格
    text = text.replace('\x00', '')      # NULL → 删除
    text = text.replace('\u200b', '')    # 零宽空格 → 删除
    text = re.sub(r'\s+', ' ', text)     # 合并多余空格
    lines = text.split("\n")
    clean_lines = [line.strip() for line in lines if line.strip()]
    return clean_lines

def preprocess_and_save(raw_dir, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = []

    for fname in sorted(os.listdir(raw_dir)):
        if fname.endswith(".txt"):
            file_path = os.path.join(raw_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                segments = clean_text(raw_text)
                print(f"{fname} 分割后得到 {len(segments)} 行")
                for seg in segments:
                    results.append({
                        "instruction": "请用儿童语言讲一个小故事：",
                        "input": "",
                        "output": seg
                    })

    with open(save_path, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ 共保存 {len(results)} 条清洗后样本到：{save_path}")

if __name__ == "__main__":
    preprocess_and_save(
        raw_dir="data/raw/arabic_ebooks_kids",  # 修改为你的原始 .txt 文件路径
        save_path="data/processed/arabic_ebooks_kids/sft_format.jsonl"
    )
