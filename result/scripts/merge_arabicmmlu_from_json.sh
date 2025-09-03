set -euo pipefail
cd "$(dirname "$0")/.."

DETAIL_OUT="results/arabicmmlu_all_models_detailed.csv"
SUMMARY_OUT="results/arabicmmlu_all_models_summary.csv"

# 统一表头（尽量贴近你之前的 CSV 结构，并额外带上 model 列）
echo "Model,Tasks,Version,Filter,n-shot,Metric,Value,Stderr" > "$DETAIL_OUT"
echo "Model,Tasks,Version,Filter,n-shot,Metric,Value,Stderr" > "$SUMMARY_OUT"

shopt -s nullglob
for f in results/*arabicmmlu*.json; do
  model_base=$(basename "$f" .json)
  nshot=$(jq -r '.config.num_fewshot // 0' "$f")
  # —— 详细：把 .results 里所有以 arabicmmlu 开头的条目都导出（每个子任务一行）
  jq -r --arg MODEL "$model_base" --argjson NSHOT "$nshot" '
    .results
    | to_entries[]
    | select(.key | startswith("arabicmmlu"))
    | . as $e
    | $e.value as $v
    | [$MODEL,        # Model
       (.key          # Tasks（原始 key）
         | sub("^arabicmmlu_?"; "")            # 去掉前缀
         | if .=="" then "arabicmmlu" else . end
         | gsub("_"; " ")                      # 用空格替下划线
       ),
       1,                                       # Version（固定 1）
       "none",                                  # Filter（固定 none）
       $NSHOT,                                  # n-shot
       "acc",                                   # Metric
       ($v.acc // $v.exact_match // 0),         # Value
       ($v.acc_stderr // $v.exact_match_stderr // 0)  # Stderr
      ]
    | @csv
  ' "$f" >> "$DETAIL_OUT"

  # —— 总结：优先使用 .groups.arabicmmlu（官方总体）；若缺失则退化为按所有 arabicmmlu* 任务的简单平均
  if jq -e '.groups.arabicmmlu.acc?' "$f" >/dev/null 2>&1; then
    jq -r --arg MODEL "$model_base" --argjson NSHOT "$nshot" '
      .groups.arabicmmlu as $g
      | [$MODEL, "arabicmmlu", 1, "none", $NSHOT, "acc", ($g.acc // 0), ($g.acc_stderr // 0)]
      | @csv
    ' "$f" >> "$SUMMARY_OUT"
  else
    jq -r --arg MODEL "$model_base" --argjson NSHOT "$nshot" '
      [ .results
        | to_entries[]
        | select(.key | startswith("arabicmmlu"))
        | .value.acc
      ] as $vals
      | if ($vals|length) > 0 then
          ($vals | add / length) as $avg
        | [$MODEL, "arabicmmlu", 1, "none", $NSHOT, "acc", $avg, 0]
        | @csv
        else empty end
    ' "$f" >> "$SUMMARY_OUT"
  fi
done
echo "Wrote: $DETAIL_OUT"
echo "Wrote: $SUMMARY_OUT"
