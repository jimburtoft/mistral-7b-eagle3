#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# EAGLE3 training on one machine: N_PROD feature producers + 1 trainer, one accelerator core / GPU each.
# Default = one trn2.3xlarge at LNC=2: producers on cores 0..2, trainer on core 3.
# Usage: train/launch.sh <run_name> <total_steps> [extra train.py args...]
# Env: ROWS (packed row glob), TARGET (target dir), WORK (~/e3_work), EVAL (eval item dir),
#      FREQ (token_freq.pt), DEVICE (auto|neuron|cuda|cpu), N_PROD (3), TRAIN_CORE (N_PROD),
#      R (rows per item, 1), NDEV (if set, process k uses device k % NDEV, e.g. NDEV=1 to share one GPU)
set -e
cd "$(dirname "$0")/.."
RUN=$1; STEPS=$2; shift 2
WORK=${WORK:-$HOME/e3_work}; DEVICE=${DEVICE:-auto}; N_PROD=${N_PROD:-3}; TRAIN_CORE=${TRAIN_CORE:-$N_PROD}; R=${R:-1}
AUX=${AUX:-2,16,29}
Q=${QUEUE_ROOT:-/dev/shm}/e3q_$RUN
OUT=$WORK/runs/$RUN
mkdir -p "$Q" "$OUT"
export NEURON_LOGICAL_NC_CONFIG=${NEURON_LOGICAL_NC_CONFIG:-2} OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
pin() { c=$1; [ -n "$NDEV" ] && c=$((c % NDEV)); export NEURON_RT_VISIBLE_CORES=$c CUDA_VISIBLE_DEVICES=$c; }
for r in $(seq 0 $((N_PROD-1))); do
  ( pin $r; exec python train/producer.py --device "$DEVICE" --rows "$ROWS" --target "$TARGET" --aux-layers "$AUX" \
      --queue "$Q" --rank $r --world $N_PROD --rows-per-item $R --max-queue ${MAXQ:-6} > "$OUT/producer$r.out" 2>&1 ) &
done
( pin $TRAIN_CORE; exec python train/train.py --device "$DEVICE" --target "$TARGET" --queue "$Q" \
    ${EVAL:+--eval-dir "$EVAL"} ${FREQ:+--token-freq "$FREQ"} --aux-layers "$AUX" --out "$OUT" --total-steps $STEPS "$@" \
    > "$OUT/trainer.out" 2>&1 ) &
TP=$!
echo "launched $RUN (trainer pid $TP); logs in $OUT"
wait $TP
for p in $(pgrep -f "train/producer.py.*$Q"); do kill $p; done
