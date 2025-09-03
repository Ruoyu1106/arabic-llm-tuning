import argparse, json, os, re, random
from collections import defaultdict
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(s=0):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def pick_label_col(levels:int)->str:
    if levels==5: return "Readability_Level_5"
    if levels==7: return "Readability_Level_7"
    if levels==19: return "Readability_Level_19"
    raise ValueError("levels must be one of {5,7,19}")

def build_prompt(sentence:str, levels:int)->str:
    # 要求只输出数字，避免出现 نعم/لا 或解释
    return (
        "قيّم مستوى سهولة القراءة للجملة التالية بالأرقام فقط من 1 إلى {K}.\n"
        "أعد الإجابة كرقم واحد فقط دون أي نص إضافي.\n"
        "الجملة:\n"
        f"{sentence}\n"
        "الإجابة: "
    ).format(K=levels)

def extract_first_int(text, lo:int, hi:int):
    m = re.search(r"(\d+)", text)
    if not m: return None
    v = int(m.group(1))
    return v if lo<=v<=hi else None

def ensure_pad(tokenizer, model):
    # 兜底 pad_token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

def load_test_split(levels:int):
    print(f"Loading BAREC test split (levels={levels})…")
    ds = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-sent", split="test")
    label_col = pick_label_col(levels)
    text_col = "Sentence"
    print(f"Loaded {len(ds)} samples, text='{text_col}', label='{label_col}'.")
    return ds, text_col, label_col

def is_jais(model_id:str)->bool:
    s = model_id.lower()
    return ("jais" in s) or ("inceptionai/jais-family" in s)

@torch.no_grad()
def eval_one_model(model_id:str, ds, text_col:str, label_col:str, levels:int, batch_size:int, dtype:str, max_length:int):
    print(f"== {model_id} ==")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    torch_dtype = {"bfloat16":torch.bfloat16,"float16":torch.float16,"float32":torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)

    # 统一设置，避免 KV-cache / padding 兼容性
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tok.padding_side = "left"
    ensure_pad(tok, model)

    total = 0
    correct = 0

    # 分级统计
    support = {l:0 for l in range(1, levels+1)}
    correct_per_level = {l:0 for l in range(1, levels+1)}
    pred_count = {l:0 for l in range(1, levels+1)}
    conf = {g:{p:0 for p in range(1,levels+1)} for g in range(1,levels+1)}

    jais_manual = is_jais(model_id)

    for i in range(0, len(ds), batch_size):
        batch = ds[i:i+batch_size]
        sentences = batch[text_col]
        labels = [int(x) for x in batch[label_col]]

        prompts = [build_prompt(s, levels) for s in sentences]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k:v.to(model.device) for k,v in enc.items()}

        if jais_manual:
            # —— JAIS 走“单步贪心”：不使用 generate()，避免 past_key_values bug
            outputs = model(**enc, use_cache=False, return_dict=True)
            next_logits = outputs.logits[:, -1, :]          # 只看下一 token
            next_ids = next_logits.argmax(dim=-1)           # 贪心
            dec = tok.batch_decode(next_ids.unsqueeze(1), skip_special_tokens=True)
        else:
            out = model.generate(
                **enc,
                max_new_tokens=6,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id
            )
            # 只解码新增部分
            dec = tok.batch_decode(out[:, enc["input_ids"].size(1):], skip_special_tokens=True)

        for pred_txt, gold in zip(dec, labels):
            pred = extract_first_int(pred_txt, 1, levels)
            support[gold] += 1
            if pred is not None:
                pred_count[pred] += 1
                conf[gold][pred] += 1
                if pred==gold:
                    correct_per_level[gold] += 1
                    correct += 1
            total += 1

        pct = correct / max(total,1)
        print(f"\r  progress {min(i+batch_size, len(ds))}/{len(ds)}  acc={pct:.4f}", end="")
    print()

    overall = correct / max(total,1)
    per_level = []
    for l in range(1, levels+1):
        s = support[l]
        c = correct_per_level[l]
        acc_l = (c / s) if s>0 else 0.0
        per_level.append({
            "level": l,
            "support": s,
            "correct": c,
            "acc": acc_l,
            "pred_count": pred_count[l],
            "pred_frac": (pred_count[l]/total if total>0 else 0.0)
        })

    return {
        "model": model_id,
        "levels": levels,
        "n_test": len(ds),
        "overall_acc": overall,
        "per_level": per_level,
        "confusion": conf
    }

def run_once(levels:int, models, batch_size:int, dtype:str, max_length:int, out_dir:str, merged_rows:list):
    ds, text_col, label_col = load_test_split(levels)
    results = []
    for mid in models:
        r = eval_one_model(mid, ds, text_col, label_col, levels, batch_size, dtype, max_length)
        print(f"{mid}: overall_acc={r['overall_acc']:.4f}")
        results.append(r)
        # 供汇总 CSV
        for row in r["per_level"]:
            merged_rows.append([
                mid, levels, r["n_test"], f"{r['overall_acc']:.4f}",
                row["level"], row["support"], row["correct"], f"{row['acc']:.4f}", row["pred_count"], f"{row['pred_frac']:.4f}"
            ])

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"barec_{levels}level_zeroshot_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"levels": levels, "n_test": len(ds), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {json_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--levels", default="5", help="5|7|19|all")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16","float32"])
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    set_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.levels == "all":
        levels_list = [5,7,19]
    else:
        lv = int(args.levels)
        if lv not in (5,7,19):
            raise ValueError("--levels must be 5|7|19|all")
        levels_list = [lv]

    merged_rows = []  # 用于合并长表
    for lv in levels_list:
        run_once(lv, args.models, args.batch_size, args.dtype, args.max_length, args.out_dir, merged_rows)

    # 输出合并 CSV（所有级别拼一起，便于一次画图）
    merged_csv = os.path.join(args.out_dir, "barec_all_levels_zeroshot_detailed.csv")
    with open(merged_csv, "w", encoding="utf-8") as f:
        f.write("model,levels,n_test,overall_acc,level,support,correct,acc,pred_count,pred_frac\n")
        for row in merged_rows:
            f.write(",".join(map(str,row)) + "\n")
    print(f"Saved -> {merged_csv}")

if __name__ == "__main__":
    main()
