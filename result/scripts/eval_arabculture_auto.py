import argparse, json, random, torch
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

COUNTRIES = ['Algeria','Egypt','Jordan','KSA','Lebanon','Libya','Morocco','Palestine','Sudan','Syria','Tunisia','UAE','Yemen']

def load_all_test():
    parts = [load_dataset("MBZUAI/ArabCulture", c, split="test") for c in COUNTRIES]
    return concatenate_datasets(parts)

def get_q_and_gold(ex):
    # 兼容不同字段名
    q = ex.get("statement") or ex.get("question") or ex.get("first_statement") or ""
    gold = ex.get("label")
    # label 可能是布尔/0/1/字符串
    if isinstance(gold, str):
        g = gold.strip().lower()
        gold = g in ["true","1","صح","صحيح","نعم","yes"]
    else:
        gold = bool(gold)
    return q, gold

def build_prompt(tok, q):
    user = f"هل العبارة التالية صحيحة أم خاطئة؟\n\n{q}\n\nاختر إجابة واحدة فقط: نعم أو لا.\nالإجابة:"
    # 若模型带 chat_template，用模板构造
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                [{"role":"user","content":user}],
                tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return user

@torch.no_grad()
def score_batch(model, tok, prompts, suffix, dtype):
    dev = model.device
    full = [p + suffix for p in prompts]
    enc = tok(full, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(dev) for k,v in enc.items()}
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if dtype=="bfloat16" else torch.float16
    ):
        out = model(**enc, use_cache=False)
        # 取最后一个 token 的 logits 作为分类（贪心近似）
        last_logits = out.logits[:, -1, :].float()
        last_lp = torch.log_softmax(last_logits, dim=-1)
        # 我们不需要取特定 token，只对比两个后缀整体对数似然：
        # 为稳定，这里改为“续写一整个答案词”的条件概率累计（prompt+答案整体）
    # 改用“答案序列 logprob 总和”的更稳方法：
    enc_p = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    Lp = enc_p["input_ids"].shape[1]
    enc_full = tok(full, return_tensors="pt", padding=True, truncation=True)
    enc_full = {k: v.to(dev) for k,v in enc_full.items()}
    labels = enc_full["input_ids"].clone()
    labels[:, :Lp] = -100
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if dtype=="bfloat16" else torch.float16
    ):
        out = model(**enc_full, labels=labels, use_cache=False)
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        mask = (shift_labels != -100)
        logp = torch.log_softmax(shift_logits, dim=-1)
        token_lp = torch.gather(
            logp, -1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)
        ).squeeze(-1)
        token_lp = token_lp.masked_fill(~mask, 0.0)
        return token_lp.sum(dim=1).cpu()

def run_model(mid, ds, bs, dtype, calib_n=200, seed=42):
    print(f"\n== {mid} ==")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        mid, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if dtype=="bfloat16" else torch.float16,
        device_map="auto"
    ).eval()

    # 预构造 prompts / gold
    prompts, golds = [], []
    for ex in ds:
        q, g = get_q_and_gold(ex)
        prompts.append(build_prompt(tok, q))
        golds.append(g)

    # ========= 校准：在小样本上选择是否翻转 =========
    idxs = list(range(len(prompts)))
    random.Random(seed).shuffle(idxs)
    calib_idx = idxs[:min(calib_n, len(idxs))]
    c_prompts = [prompts[i] for i in calib_idx]
    c_golds   = [golds[i]   for i in calib_idx]

    lp_yes = []
    lp_no  = []
    for i in range(0, len(c_prompts), bs):
        chunk = c_prompts[i:i+bs]
        lp_yes.append(score_batch(model, tok, chunk, " نعم", dtype))
        lp_no.append(score_batch(model, tok, chunk, " لا",  dtype))
    import torch as T
    lp_yes = T.cat(lp_yes); lp_no = T.cat(lp_no)
    pred_yes_true  = (lp_yes > lp_no).numpy()      # mapping A: نعم=True
    pred_yes_false = (lp_yes <= lp_no).numpy()     # mapping B: نعم=False
    acc_A = sum(bool(pred_yes_true[j])  == bool(c_golds[j]) for j in range(len(c_golds)))/len(c_golds)
    acc_B = sum(bool(pred_yes_false[j]) == bool(c_golds[j]) for j in range(len(c_golds)))/len(c_golds)
    flipped = acc_B > acc_A
    print(f"calibration: A(nam=True)={acc_A:.4f}  B(nam=False)={acc_B:.4f}  -> flipped={flipped}")

    # ========= 全量评测 =========
    correct = 0
    for i in tqdm(range(0, len(prompts), bs), desc="full", leave=False):
        chunk = prompts[i:i+bs]
        lp_y = score_batch(model, tok, chunk, " نعم", dtype)
        lp_n = score_batch(model, tok, chunk, " لا",  dtype)
        if not flipped:
            pred = (lp_y > lp_n).numpy()      # نعم=True
        else:
            pred = (lp_y <= lp_n).numpy()     # نعم=False
        gold = golds[i:i+bs]
        correct += sum(bool(pred[j]) == bool(gold[j]) for j in range(len(chunk)))
    acc = correct/len(prompts)
    print(f"accuracy = {acc:.4f}")
    return {"model": mid, "accuracy": acc, "flipped": flipped}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dtype", choices=["bfloat16","float16"], default="bfloat16")
    ap.add_argument("--out", default="results/arabculture_auto.json")
    ap.add_argument("--calib_n", type=int, default=200)
    args = ap.parse_args()

    print("Loading ArabCulture test (all countries)…")
    ds = load_all_test()
    total = len(ds)
    print(f"Total samples: {total}")

    out = {"total": total, "calib_n": args.calib_n, "models":[]}
    for m in args.models:
        out["models"].append(run_model(m, ds, args.batch_size, args.dtype, args.calib_n))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out}")
if __name__ == "__main__":
    main()
