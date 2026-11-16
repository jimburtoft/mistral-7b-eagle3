#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Benchmark a running server on a JSONL prompt set ({"prompt": ...} per line) at given concurrencies.
# Usage: eval/serving/bench.sh <tag> <tokenizer> <prompts.jsonl> <concurrency...>
# Env: OUT (256 output tokens), RESULTS (./results), PORT (8000)
set -euo pipefail
TAG=$1 TOK=$2 DS=$3; shift 3
R=${RESULTS:-results}; mkdir -p "$R"
for C in "$@"; do
  N=$(( C * 8 )); [ $N -lt 16 ] && N=16
  vllm bench serve --base-url http://localhost:${PORT:-8000} --model m --tokenizer "$TOK" \
    --dataset-name custom --dataset-path "$DS" --custom-output-len ${OUT:-256} \
    --backend openai-chat --endpoint /v1/chat/completions --seed 0 \
    --num-prompts $N --max-concurrency $C --temperature 0 --ignore-eos \
    --save-result --result-dir "$R" --result-filename $TAG-c$C.json > "$R/$TAG-c$C.txt" 2>&1 || tail -20 "$R/$TAG-c$C.txt"
  grep -E "Output token throughput|Median TPOT|Acceptance length" "$R/$TAG-c$C.txt" | sed "s/^/[$TAG c=$C] /" || true
done
