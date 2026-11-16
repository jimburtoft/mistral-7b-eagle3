#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serve the target (optionally with the EAGLE3 draft) with vLLM.
# Usage: eval/serving/serve.sh <model> [draft] [k]
# Env: PLATFORM=neuron|gpu (default neuron), TP (4 on a trn2.3xlarge, 1 on one GPU), MML (2048), MNS (8),
#      PORT (8000), OUT (log dir, ./logs)
# Neuron: run inside the vLLM-Neuron environment (vllm + vllm-neuron); set PYTHONPATH to a vllm-neuron
#         build that supports MistralForCausalLM if your installed release does not.
set -euo pipefail
MODEL=$1 DRAFT=${2:-} K=${3:-3}
PLATFORM=${PLATFORM:-neuron} MML=${MML:-2048} MNS=${MNS:-8} OUT=${OUT:-logs}
mkdir -p "$OUT"
ARGS=(--served-model-name m --port "${PORT:-8000}" --max-model-len "$MML" --max-num-seqs "$MNS" --no-enable-prefix-caching)
if [ -n "$DRAFT" ]; then
  ARGS+=(--speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": $K}")
fi
if [ "$PLATFORM" = neuron ]; then
  export NEURON_SKIP_EFA_AFFINITY=1   # trn2.3xlarge has no EFA device
  ARGS+=(--tensor-parallel-size "${TP:-4}" --max-num-batched-tokens "$MML"
         --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}}}')
else
  ARGS+=(--tensor-parallel-size "${TP:-1}")
fi
exec vllm serve "$MODEL" "${ARGS[@]}" > "$OUT/serve.log" 2>&1
