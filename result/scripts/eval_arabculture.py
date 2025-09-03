import argparse, json, math, os, sys, random
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# 数据加载：ArabCulture（13个config）
# ---------------------------
COUNTRIES = ["Algeria","Egypt","Jordan","KSA","Lebanon","Libya",
             "Morocco","Palestine","Sudan","Syria","Tunisia","UAE","Yemen"]

def load_all_countries_test() -> Dict[str, "datasets.Dataset"]:
    ds_map = {}
    for c in COUNTRIES:
        ds = load_dataset("MBZUAI/ArabCulture", c, split="test")
        ds_map[c] = ds
    return ds_map

def concat_all(ds_map: Dict[str, "datasets.Dataset"]):
    return concatenate_datasets([ds_map[c] for c in COUNTRIES])

# ---------------------------
# 样本规范化（尽量兼容多种字段名）
# 目标：得到 (prompt_text, gold_label[0/1])
# gold=1 表示“是/正确/True”，gold=0 表示“否/错误/False”
# ---------------------------
def extract_text_and_label(ex):
    # 文本/问题
    text = (
        ex.get("question")
        or ex.get("prompt")
        or ex.get("text")
        or ex.get("first_statement")
        or ex.get("sentence")
        or ex.get("statement")
        or ""
    )

    # 如果有 second_statement，就把它拼进来做蕴含/一致性判断的提示
    if ex.get("second_statement"):
        text = f"{ex.get('first_statement','').strip()}\n{ex.get('second_statement','').strip()}"

    # 标签名兼容
    if "label" in ex:
        lab = ex["label"]
    elif "gold" in ex:
        lab = ex["gold"]
    elif "answer" in ex:
        # 把常见答案映射到 0/1
        ans = str(ex["answer"]).strip().lower()
        if ans in ["1","true","صح","صحيح","نعم","yes","y","true."]:
            lab = 1
        elif ans in ["0","false","خطأ","لا","no","n","false."]:
            lab = 0
        else:
            # 不可识别，默认 -1
            lab = -1
    else:
        lab = -1

    # 有些数据会用 bool、"yes"/"no"
    if isinstance(lab, bool):
        lab = 1 if lab else 0
    if isinstance(lab, str):
        low = lab.strip().lower()
        if low in ["1","true","صح","صحيح","نعم","yes","y"]:
            lab = 1
        elif low in ["0","false","خطأ","لا","no","n"]:
            lab = 0
        else:
            lab = -1

    return text, lab

# ---------------------------
# 评分：对“ نعم / لا ”与“ صح / خطأ ”两套答案做对数似然对比
#   我们只用一步 next-token 的条件概率：P(choice_token | prompt)
#   gold=1 期望选择 yes/صح，gold=0 期望选择 no/خطأ
# ---------------------------
CHOICE_SETS = [
    (" نعم", " لا"),
    (" صح", " خطأ"),
]

def prepare_batch(tokenizer, prompts: List[str], max_length: int = 1024):
    # 截断到模型可接受的长度；右截断更安全
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

@torch.no_grad()
def score_choice_next_token(model, tokenizer, input_ids, attention_mask, choice_str: str):
    """
    对每个样本，计算把 choice_str 作为下一个 token 的 logprob。
    注意：choice_str 可能被切成多个 token，这里只用第一个 token 的logprob，
    实际上足以稳定地区分 是/否 这种短答案。
    """
    device = model.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 前向得到最后位置的 logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]
    # 每个样本的最后一个非pad位置
    last_index = attention_mask.sum(dim=1) - 1  # [B]
    last_logits = logits[torch.arange(logits.size(0), device=device), last_index]  # [B, V]
    logprobs = torch.log_softmax(last_logits, dim=-1)  # [B, V]

    # choice 第一个token
    choice_ids = tokenizer(choice_str, add_special_tokens=False).input_ids
    if len(choice_ids) == 0:
        # 如果异常，就给极低分
        return torch.full((input_ids.size(0),), -1e9, device=device)

    first_token_id = choice_ids[0]
    return logprobs[:, first_token_id]  # [B]

@torch.no_grad()
def batch_predict(model, tokenizer, texts: List[str], batch_size: int = 32, max_length: int = 1024):
    """
    返回：每个 choice_set 对应的 (logprob_yes, logprob_no)
    形状：len(CHOICE_SETS) 个条目，每个条目是一个 (yes_logps, no_logps) tuple，二者都是 [N] Tensor
    """
    all_yes = [torch.empty(0, device=model.device) for _ in CHOICE_SETS]
    all_no  = [torch.empty(0, device=model.device) for _ in CHOICE_SETS]

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        batch = prepare_batch(tokenizer, chunk, max_length=max_length)
        # 逐个 choice_set 评分
        for idx, (yes_tok, no_tok) in enumerate(CHOICE_SETS):
            y_lp = score_choice_next_token(model, tokenizer, batch["input_ids"], batch["attention_mask"], yes_tok)
            n_lp = score_choice_next_token(model, tokenizer, batch["input_ids"], batch["attention_mask"], no_tok)
            all_yes[idx] = torch.cat([all_yes[idx], y_lp], dim=0)
            all_no[idx]  = torch.cat([all_no[idx],  n_lp], dim=0)

    return list(zip(all_yes, all_no))  # [(yes_logps[N], no_logps[N]), ...]

def accuracy_from_logits(yes_lp, no_lp, gold: List[int], flipped: bool):
    """
    如果 flipped=False： gold=1 选 (yes_lp>no_lp)；gold=0 选 (no_lp>yes_lp)
    如果 flipped=True ： gold=1 选 (no_lp>yes_lp)；gold=0 选 (yes_lp>no_lp)
    """
    yes_better = (yes_lp > no_lp)
    preds = []
    for i, g in enumerate(gold):
        if g not in (0,1):
            preds.append(-1)  # 跳过无效
            continue
        if not flipped:
            pred = 1 if yes_better[i].item() else 0
        else:
            pred = 0 if yes_better[i].item() else 1
        preds.append(pred)
    # 计算准确率（忽略无效标签）
    ok, tot = 0, 0
    for p, g in zip(preds, gold):
        if g in (0,1) and p in (0,1):
            ok += int(p == g)
            tot += 1
    return ok / max(1, tot)

def calibrate_flip(model, tokenizer, texts: List[str], gold: List[int], calib_n: int, batch_size: int):
    """
    在前 calib_n 个样本上，尝试两套 choice，并判断 flipped True/False 哪个更好。
    返回：(best_choice_idx, flipped_bool)
    """
    n = min(calib_n, len(texts))
    texts_c = texts[:n]
    gold_c  = gold[:n]
    all_scores = batch_predict(model, tokenizer, texts_c, batch_size=batch_size)
    best_acc = -1.0
    best_choice = 0
    best_flip = False
    for idx, (yes_lp, no_lp) in enumerate(all_scores):
        for flipped in (False, True):
            acc = accuracy_from_logits(yes_lp, no_lp, gold_c, flipped)
            if acc > best_acc:
                best_acc = acc
                best_choice = idx
                best_flip = flipped
    return best_choice, best_flip, best_acc

def eval_on_dataset(model, tokenizer, texts: List[str], gold: List[int],
                    batch_size: int, calib_n: int, dtype_str: str):
    # 校准
    choice_idx, flipped, calib_acc = calibrate_flip(model, tokenizer, texts, gold, calib_n, batch_size)
    # 全量评估
    yes_lp, no_lp = batch_predict(model, tokenizer, texts, batch_size=batch_size)[choice_idx]
    overall_acc = accuracy_from_logits(yes_lp, no_lp, gold, flipped)
    return overall_acc, flipped, choice_idx, calib_acc

# ---------------------------
# 主流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="list of HF model ids")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16","float32"])
    parser.add_argument("--calib_n", type=int, default=200)
    parser.add_argument("--per_country", action="store_true", help="also output per-country scores")
    parser.add_argument("--out", type=str, required=True, help="output json path")
    args = parser.parse_args()

    print("Loading ArabCulture test (all countries)…")
    ds_map = load_all_countries_test()
    ds_all = concat_all(ds_map)
    total = len(ds_all)
    print(f"Total samples: {total}")

    # 先把 all 的文本和标签提取出来
    texts_all, gold_all = [], []
    for ex in ds_all:
        t, g = extract_text_and_label(ex)
        texts_all.append(t)
        gold_all.append(g)

    # 如果需要 per_country，也先准备好每个国家的文本和标签
    per_country_data = {}
    if args.per_country:
        for c in COUNTRIES:
            tt, gg = [], []
            for ex in ds_map[c]:
                t, g = extract_text_and_label(ex)
                tt.append(t)
                gg.append(g)
            per_country_data[c] = (tt, gg)

    # dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    out_obj = {
        "total": total,
        "calib_n": args.calib_n,
        "models": []
    }

    for m in args.models:
        print(f"\n== {m} ==")
        tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            m,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        model.eval()

        # overall
        overall_acc, flipped, choice_idx, calib_acc = eval_on_dataset(
            model, tok, texts_all, gold_all,
            batch_size=args.batch_size, calib_n=args.calib_n, dtype_str=args.dtype
        )
        print(f"calibration: choice={CHOICE_SETS[choice_idx]}  flipped={flipped}  calib_acc={calib_acc:.4f}")
        print(f"accuracy = {overall_acc:.4f}")

        entry = {
            "model": m,
            "accuracy": float(overall_acc),
            "flipped": bool(flipped),
            "choice": CHOICE_SETS[choice_idx],
            "calib_acc": float(calib_acc),
        }

        # per-country
        if args.per_country:
            pc = {}
            for c in COUNTRIES:
                tt, gg = per_country_data[c]
                acc_c, _, _, _ = eval_on_dataset(model, tok, tt, gg,
                                                 batch_size=args.batch_size, calib_n=min(args.calib_n, len(tt)), dtype_str=args.dtype)
                pc[c] = float(acc_c)
                print(f"  {c}: {acc_c:.4f}")
            entry["per_country"] = pc

        out_obj["models"].append(entry)

        # 释放显存
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
