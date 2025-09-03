import argparse, json, torch
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

COUNTRIES = ['Algeria','Egypt','Jordan','KSA','Lebanon','Libya','Morocco','Palestine','Sudan','Syria','Tunisia','UAE','Yemen']

def load_arabculture_all_test(max_samples=None):
    parts = []
    for c in COUNTRIES:
        ds = load_dataset("MBZUAI/ArabCulture", c, split="test")
        parts.append(ds)
    ds = concatenate_datasets(parts)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds

def normalize(ex):
    q = ex.get("statement") or ex.get("question") or ex.get("first_statement") or ""
    gold = ex.get("label")
    if isinstance(gold, str):
        gold = gold.strip().lower()
        gold = True if gold in ["true","1","صح","صحيح","نعم"] else False
    else:
        gold = bool(gold)
    return q, gold

def build_prompt(tokenizer, question):
    user = f"هل العبارة التالية صحيحة أم خاطئة؟\n\n{question}\n\nاختر إجابة واحدة فقط: صحيح أو خطأ.\nالإجابة:"
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template([{"role":"user","content":user}],
                                                tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return user

@torch.no_grad()
def score_batch(model, tokenizer, prompts, answer_str, dtype):
    # 返回每个样本的“答案串”总 logprob（逐 token 累加）
    device = model.device
    # 先独立编码 prompt 拿长度
    enc_p = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    Lp = enc_p["input_ids"].shape[1]
    # 再编码 prompt+answer
    full = [p + answer_str for p in prompts]
    enc = tokenizer(full, return_tensors="pt", padding=True, truncation=True)
    for k in enc: enc[k] = enc[k].to(device)
    # 构造 labels：只计算答案区域（prompt 长度到句尾）
    labels = enc["input_ids"].clone()
    labels[:, :Lp] = -100
    with torch.autocast(device_type="cuda",
                        dtype=torch.bfloat16 if dtype=="bfloat16" else torch.float16):
        out = model(**enc, labels=labels)
        # CE loss 是 batch 平均到“有效 token”上的均值；我们要还原到逐样本“有效 token 总和”
        # 方案：逐 token loss（使用 logits -> per-token NLL）
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        # mask 有效 token
        mask = (shift_labels != -100)
        # 取标签对应的 logsoftmax
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = torch.gather(log_probs, -1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp.masked_fill(~mask, 0.0)
        # 按样本求和
        sample_lp = token_lp.sum(dim=1).cpu()
    return sample_lp

def eval_one_model(model_id, ds, bs, dtype):
    print(f"\nEvaluating {model_id} …", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # 更健壮的空格侧与 padding 侧
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if dtype=="bfloat16" else torch.float16,
        device_map="auto"
    ).eval()

    TRUE_FALSE_SETS = [
        ("space_true_false", " صحيح", " خطأ"),
        ("no_space_true_false", "صحيح", "خطأ"),
        ("space_sah_khata", " صح", " خطأ"),
        ("sah_khata", "صح", "خطأ"),
        ("naam_laa_space", " نعم", " لا"),
        ("naam_laa", "نعم", "لا"),
    ]

    prompts, golds = [], []
    for ex in ds:
        q, g = normalize(ex)
        prompts.append(build_prompt(tok, q))
        golds.append(g)

    results = []
    for name, t_str, f_str in TRUE_FALSE_SETS:
        correct = 0
        for i in tqdm(range(0, len(prompts), bs), desc=f"{name}", leave=False):
            chunk = prompts[i:i+bs]
            lp_t = score_batch(model, tok, chunk, t_str, dtype)
            lp_f = score_batch(model, tok, chunk, f_str, dtype)
            pred_true = (lp_t > lp_f).numpy()
            gold_chunk = golds[i:i+bs]
            for j, pred in enumerate(pred_true):
                if bool(pred) == bool(gold_chunk[j]):
                    correct += 1
        acc = correct/len(prompts)
        results.append((name, acc, False))
        # flipped sanity
        results.append((name, 1.0-acc, True))

    best = max(results, key=lambda x: x[1])
    # 打印小表
    print(f"{model_id}: best={best[1]:.4f} with [{best[0]}{' FLIPPED' if best[2] else ''}]")
    print("| setting | flipped? | accuracy |")
    print("|---|:---:|---:|")
    for n,a,f in results:
        print(f"| {n} | {'yes' if f else 'no'} | {a:.4f} |")

    return {
        "model": model_id,
        "best_accuracy": best[1],
        "best_setting": best[0],
        "flipped": best[2],
        "detail": [{"setting":n,"flipped":f,"accuracy":a} for n,a,f in results]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dtype", choices=["bfloat16","float16"], default="bfloat16")
    ap.add_argument("--max_samples", type=int, default=0, help="先冒烟测试时可设 200 等")
    ap.add_argument("--out", default="results/arabculture_diagnose_fast.json")
    args = ap.parse_args()

    print("Loading ArabCulture (all countries, test)…")
    ds = load_arabculture_all_test(args.max_samples if args.max_samples>0 else None)
    print(f"Total samples: {len(ds)}")

    out = {"total": len(ds), "models":[]}
    torch.set_grad_enabled(False)

    for m in args.models:
        out["models"].append(eval_one_model(m, ds, args.batch_size, args.dtype))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out}")
if __name__ == "__main__":
    main()
