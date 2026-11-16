#!/usr/bin/env bash
# CPU end-to-end smoke: pack -> token freq -> producer (tiny random Mistral, real tokenizer) ->
# eval items -> trainer (few steps) -> speculators checkpoint -> reload + loss check.
# Usage: tests/cpu_smoke.sh <python> <regenerated.jsonl> <tokenizer_dir>
set -euo pipefail
PY=$1 INP=$2 TOK=$3
W=${W:-tmp/smoke}
rm -rf "$W"; mkdir -p "$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
head -n 24 "$INP" > "$W/regen.jsonl"
$PY data/pack.py --inp "$W/regen.jsonl" --tokenizer "$TOK" --out "$W/rows" --seq-len 512 --rows-per-shard 8
$PY data/token_freq.py --rows "$W/rows/rows_*.npz" --out "$W/token_freq.pt"
$PY - "$W/tiny" "$TOK" <<'EOF'
import sys, torch
from transformers import MistralConfig, MistralForCausalLM
out, tok = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
c = MistralConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=8, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, vocab_size=32768, rope_theta=1e6, sliding_window=None,
                  max_position_embeddings=4096, tie_word_embeddings=False)
MistralForCausalLM(c).save_pretrained(out, safe_serialization=True)
EOF
$PY train/producer.py --device cpu --rows "$W/rows/rows_*.npz" --target "$W/tiny" --aux-layers 1,4,5 \
  --eval-out "$W/eval" --rows-per-item 2
$PY train/producer.py --device cpu --rows "$W/rows/rows_*.npz" --target "$W/tiny" --aux-layers 1,4,5 \
  --queue "$W/q" --rows-per-item 2 --epochs 1 --max-queue 100
ls "$W/q" | head -3
$PY train/train.py --device cpu --target "$W/tiny" --queue "$W/q" --eval-dir "$W/eval" --out "$W/run" \
  --total-steps 4 --aux-layers 1,4,5 --draft-vocab 2048 --token-freq "$W/token_freq.pt" --lr 1e-3 --warmup 1 \
  --log-every 1 --eval-every 4 --save-every 4 --keep-queue
$PY - "$W" <<'EOF'
import sys, json, torch
sys.path.insert(0, ".")
from pathlib import Path
from eagle3.checkpoint import load_speculators
from train.train import Step, evaluate
W = Path(sys.argv[1])
d = load_speculators(str(W / "run/step0000004"))
cfg = json.load(open(W / "run/step0000004/config.json"))
print("reloaded:", cfg["architectures"], "draft_vocab", cfg["draft_vocab_size"], "aux", cfg["eagle_aux_hidden_state_layer_ids"])
r = evaluate(Step(d.float(), 3, True), W / "eval", torch.device("cpu"))
last = [json.loads(l) for l in open(W / "run/eval.jsonl")][-1]
print("reloaded eval", r, "\ntrainer eval ", {k: last[k] for k in ("loss", "accept_len")})
assert abs(r["loss"] - last["loss"]) < 2e-2 * max(1, last["loss"]), "reloaded checkpoint loss mismatch"
print("CPU SMOKE OK")
EOF
$PY eval/acceptance.py --device cpu --draft "$W/run/step0000004" --target "$W/tiny" --rows "$W/rows/rows_0000.npz" --k 3
echo "EVAL SMOKE OK"
