#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BAREC 多级可读性单步评测（适配 decoder-only & JAIS）
- 数据：CAMeL-Lab/BAREC-Corpus-v1.0（test）
- 级别：--levels 3,5,7,19 或 --levels all
- 评测：单步前向，直接用最后一步 logits 选数字（支持 '1'..'9' 与 '١'..'٩'）
- 输出：
  * results/barec_full_overall.csv
  * results/barec_full_{model}_level{K}_by_group.csv
  * results/barec_full_{model}_level{K}_by_len.csv
  * results/barec_full_{model}_level{K}_details.json
"""

import os
import re
import json
import argparse
import math
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------
# 实用函数
# -----------------------

def sanitize_fn(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def get_level_field(k: int) -> str:
    # BAREC 提供 Readability_Level_3/5/7/19
    return f"Readability_Level_{k}"

def bucket_by_len(word_counts):
    # 三档：短(0-33%) / 中(33-66%) / 长(66-100%)
    t = torch.tensor(word_counts, dtype=torch.float32)
    q1 = torch.quantile(t, 0.33).item()
    q2 = torch.quantile(t, 0.66).item()
    cuts = (max(1, int(q1)), max(1, int(q2)))
    return cuts

def arabic_digits_for_K(K):
    latin = [str(i) for i in range(1, K+1)]
    arabic_indic_map = {
        1:"١",2:"٢",3:"٣",4:"٤",5:"٥",
        6:"٦",7:"٧",8:"٨",9:"٩"
    }
    combos = []
    for i in range(1, K+1):
        alt = arabic_indic_map.get(i)
        if alt:
            combos.append({latin[i-1], alt})
        else:
            combos.append({latin[i-1]})
    return combos

def make_label_token_ids(tokenizer, K):
    # 为每一类收集可能的一字 token 的 id（包含 ' 1', '1', 前有空格的情况更稳）
    combos = arabic_digits_for_K(K)
    label_token_ids = []
    candidates_cache = {}
    def ids_for(txt):
        if txt in candidates_cache:
            return candidates_cache[txt]
        # 尝试裸字与带空格两种
        ids = set()
        for s in [txt, " " + txt]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        candidates_cache[txt] = ids
        return ids

    for cset in combos:
        ids = set()
        for ch in cset:
            ids |= ids_for(ch)
        label_token_ids.append(sorted(list(ids)))
    return label_token_ids  # length K，元素为若干 token id

def logits_to_class(logits_last, label_token_ids):
    # logits_last: [V]
    # 对每个类别，把对应 token 的 logit 做 logsumexp（若只有一个就取该 logit）
    K = len(label_token_ids)
    scores = []
    for i in range(K):
        ids = label_token_ids[i]
        if not ids:
            scores.append(float("-inf"))
        else:
            vals = logits_last[ids]  # [m]
            # 避免数值问题
            m = vals.max()
            score = m + torch.log(torch.exp(vals - m).sum())
            scores.append(score.item())
    return int(torch.tensor(scores).argmax().item()) + 1, scores

def load_barec_test():
    print("Loading BAREC test split…")
    ds = load_dataset("CAMeL-Lab/BAREC-Corpus-v1.0", split="test")
    # 关键字段
    fields = ds.features.keys()
    print(f"Loaded {len(ds)} samples; fields={list(fields)}")
    return ds

def build_prompt(sentence, K):
    # 简短直接：让模型输出 1..K 的单个数字
    # 说明：不用 few-shot，不用生成，只取下一 token 的数字即可
    return (
        f"قيّم قابلية قراءة الجملة التالية على مقياس من 1 إلى {K} "
        f"(1=الأبسط، {K}=الأصعب). اكتب رقماً واحداً فقط.\n"
        f"الجملة: {sentence}\n"
        f"التصنيف:"
    )

def load_model_and_tokenizer(mid, dtype, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    # decoder-only 要左填充，且 pad_token 兜底
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    # 保险：对于 JAIS 等模型，强制 use_cache=True 但我们不会用 generate()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    return tok, model

def single_step_predict_batch(tok, model, prompts, label_token_ids, max_length=1024):
    # returns: preds(list[int]), confs(list[float])（最大类的 softmax 概率）
    # 纯前向，不用 generate
    enc = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        # 取每条的最后一个位置的 logits
        # 注意：left padding -> 最后 token 索引需要考虑各自长度
        logits = out.logits  # [B, T, V]
        B, T, V = logits.shape
        preds = []
        confs = []
        for i in range(B):
            # 找到有效长度
            attn = enc["attention_mask"][i]  # [T]
            last_idx = int(attn.nonzero()[-1].item())
            last_logits = logits[i, last_idx, :]  # [V]
            cls, scores = logits_to_class(last_logits, label_token_ids)
            # 计算 softmax 概率
            s = torch.tensor(scores)
            prob = torch.softmax(s, dim=-1)[cls-1].item()
            preds.append(cls)
            confs.append(prob)
    return preds, confs

def evaluate_one_level(mid, tok, model, ds, K, group_field, out_dir, max_length=1024, batch_size=48):
    label_field = get_level_field(K)
    y_true = []
    y_pred = []
    y_conf = []
    word_counts = []

    # label tokens
    label_token_ids = make_label_token_ids(tok, K)

    # 准备数据
    samples = []
    for ex in ds:
        # 只保留 1..K 范围内的
        y = int(ex[label_field])
        if 1 <= y <= K:
            samples.append({
                "text": ex["Sentence"],
                "y": y,
                "word_count": int(ex["Word_Count"]),
                "group": str(ex.get(group_field, "NA")) if group_field else "ALL"
            })

    if not samples:
        return None

    # 评测
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i+batch_size]
        prompts = [build_prompt(s["text"], K) for s in chunk]
        preds, confs = single_step_predict_batch(tok, model, prompts, label_token_ids, max_length=max_length)
        for j, s in enumerate(chunk):
            y_true.append(s["y"])
            y_pred.append(int(preds[j]))
            y_conf.append(float(confs[j]))
            word_counts.append(s["word_count"])

    # 汇总：整体
    correct = sum(int(a==b) for a,b in zip(y_true,y_pred))
    acc = correct/len(y_true)
    # 预测分布与混淆矩阵
    pred_dist = Counter(y_pred)
    conf_mat = defaultdict(lambda: Counter())
    for a,b in zip(y_true,y_pred):
        conf_mat[a][b]+=1
    conf_mat = {str(k): dict(v) for k,v in conf_mat.items()}

    # 分组：group_field
    by_group = defaultdict(lambda: {"n":0, "correct":0})
    for t,p,s in zip(y_true,y_pred,samples):
        g = s["group"]
        by_group[g]["n"] += 1
        by_group[g]["correct"] += int(t==p)
    by_group_rows = []
    for g,stat in sorted(by_group.items(), key=lambda x: (-x[1]["n"], x[0])):
        g_acc = stat["correct"]/stat["n"] if stat["n"] else 0.0
        by_group_rows.append([g, stat["n"], round(g_acc,4)])

    # 长短句：按 word_count 三分位
    wc_cuts = bucket_by_len(word_counts)
    def wc_bucket(w):
        if w <= wc_cuts[0]: return "short"
        elif w <= wc_cuts[1]: return "medium"
        else: return "long"
    by_len = defaultdict(lambda: {"n":0, "correct":0})
    for (t,p,w) in zip(y_true,y_pred,word_counts):
        b = wc_bucket(w)
        by_len[b]["n"] += 1
        by_len[b]["correct"] += int(t==p)
    by_len_rows = []
    for b in ["short","medium","long"]:
        n = by_len[b]["n"]
        a = by_len[b]["correct"]/n if n else 0.0
        by_len_rows.append([b, n, round(a,4)])

    # 写文件
    base = f"{sanitize_fn(mid)}_level{K}"
    # 详细 JSON
    details = {
        "model": mid,
        "level": K,
        "support": len(y_true),
        "accuracy": acc,
        "pred_dist": dict(pred_dist),
        "confusion_matrix": conf_mat,
        "group_field": group_field or "ALL",
        "by_group": {r[0]: {"n": r[1], "acc": r[2]} for r in by_group_rows},
        "by_len": {r[0]: {"n": r[1], "acc": r[2]} for r in by_len_rows},
        "wc_cuts": {"short<= ": wc_cuts[0], "medium<= ": wc_cuts[1]}
    }
    with open(os.path.join(out_dir, f"barec_full_{base}_details.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    # by_group CSV
    with open(os.path.join(out_dir, f"barec_full_{base}_by_group.csv"), "w", encoding="utf-8") as f:
        f.write("group,n,acc\n")
        for r in by_group_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f}\n")

    # by_len CSV
    with open(os.path.join(out_dir, f"barec_full_{base}_by_len.csv"), "w", encoding="utf-8") as f:
        f.write("bucket,n,acc\n")
        for r in by_len_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f}\n")

    return {
        "model": mid,
        "level": K,
        "support": len(y_true),
        "accuracy": acc
    }

# -----------------------
# 主流程
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="HuggingFace repo ids, e.g. tiiuae/Falcon3-7B-Instruct ..."
    )
    parser.add_argument(
        "--levels", type=str, default="5",
        help="e.g. '3,5,7,19' or 'all'"
    )
    parser.add_argument(
        "--batch_size", type=int, default=48
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16","float16","float32"]
    )
    parser.add_argument(
        "--max_length", type=int, default=1024
    )
    parser.add_argument(
        "--group_field", type=str, default="Text_Class",
        help="可用：Text_Class / Domain / Author / Source ... 或留空"
    )
    parser.add_argument(
        "--out_prefix", type=str, default="results/barec_full",
        help="输出前缀（目录会自动创建）"
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_prefix)
    if not out_dir:
        out_dir = "results"
    ensure_dir(out_dir)

    # 解析 levels
    if args.levels.strip().lower() == "all":
        levels = [3,5,7,19]
    else:
        levels = [int(x) for x in re.split(r"[,\s]+", args.levels.strip()) if x]

    # 加载数据
    ds = load_barec_test()

    # 汇总 overall
    overall_rows = [["model","level","support","accuracy"]]

    for mid in args.models:
        print(f"\n=== Evaluating {mid} ===")
        tok, model = load_model_and_tokenizer(mid, args.dtype)

        for K in levels:
            print(f"  Level-{K} classification...")
            res = evaluate_one_level(
                mid, tok, model, ds, K,
                group_field=args.group_field if args.group_field else None,
                out_dir=out_dir,
                max_length=args.max_length,
                batch_size=args.batch_size
            )
            if res is None:
                print(f"  [WARN] No valid samples for level-{K}")
                continue
            overall_rows.append([
                mid, str(K), str(res["support"]), f"{res['accuracy']:.4f}"
            ])
            print(f"  -> acc={res['accuracy']:.4f} (n={res['support']})")

    # 写 overall CSV
    overall_csv = args.out_prefix + "_overall.csv"
    with open(overall_csv, "w", encoding="utf-8") as f:
        for row in overall_rows:
            f.write(",".join(row) + "\n")
    print(f"\nSaved overall -> {overall_csv}")

if __name__ == "__main__":
    main()
