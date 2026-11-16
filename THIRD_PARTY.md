# Third-party code and models

All code dependencies are public, OSI-licensed open-source projects.

| Component | Source (version / commit) | License | How used |
|---|---|---|---|
| speculators | <https://github.com/vllm-project/speculators> v0.9.0 (`1c800f2b053bcd538fc11bf49ec843d4dfa81a95`) | Apache-2.0 | Reference for the EAGLE3 draft architecture, training objective (TTT steps, KL loss), draft-vocab mapping and checkpoint format. Re-implemented here in plain PyTorch (no source files copied); `tests/parity_speculators.py` imports it at test time to check loss and gradients. |
| transformers | <https://github.com/huggingface/transformers> (>= 5.17) | Apache-2.0 | Tokenizer / chat template; reference `MistralForCausalLM` in `tests/parity_target.py`. The target forward used for training is our own functional implementation (`eagle3/target.py`). |
| PyTorch | <https://github.com/pytorch/pytorch> | BSD-3-Clause | Training framework (CPU / CUDA / Neuron devices). |
| torch-neuronx | AWS Neuron SDK 2.33 public DLAMI venv `aws_neuronx_venv_torch_neuronx_2_14_0` | Apache-2.0 (public SDK) | Trainium device + `torch.compile(backend="neuron")`. Optional: not needed on GPU / CPU. |
| vLLM | <https://github.com/vllm-project/vllm> 0.29.0 | Apache-2.0 | Serving + `vllm bench serve` (GPU or Neuron). |
| vllm-neuron | <https://github.com/vllm-project/vllm-neuron> 0.29.0.1.2.0 and public fork <https://github.com/jimburtoft/vllm-neuron> branch `mistral-trn2-0.29` | Apache-2.0 | Trainium serving of the target (+ EAGLE3 draft) for response regeneration and evaluation. Mistral support was added in the public fork. |
| datasets | <https://github.com/huggingface/datasets> | Apache-2.0 | Loading prompt datasets. |

| Model | Source (revision) | License | Use |
|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` (`c170c708c41dac9275d15a8fff4eca08d52bab71`) | Apache-2.0 | Target / verifier |

Datasets: see `data/LICENSES.md`.
