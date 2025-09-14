# VLoRP: Various‑Grained Low‑Rank Gradient Projection (with ProjFactor)

A lightweight, memory‑efficient fine‑tuning toolkit that projects gradients into a low‑dimensional subspace at **adjustable granularity** and updates with an Adam‑like optimizer **ProjFactor**.

> **What’s new?** VLoRP adds a *projection granularity* hyperparameter `c` in addition to the usual rank `r`, letting you trade memory ↔ performance under a fixed budget \(M = c·r\). Finer granularity (larger `c`, smaller `r`) consistently yields stronger results at the same memory cost, and the method introduces **no extra compute** beyond a pair of reshape ops.  


---

## Key Ideas

- **Gradient projection with granularity.** Reshape each parameter’s gradient matrix \(G ∈ ℝ^{n×m}\) to \(\tilde G ∈ ℝ^{nc×(m/c)}\), project rows with a random matrix \(\tilde P ∈ ℝ^{(m/c)×r}\), store \(\tilde G_s=\tilde G\tilde P\), then project‑back and reshape for the update. Only reshapes are added; compute overhead is unchanged. 
- **Two update schemes, OS > SS.** Original‑space (OS) adapts in the full space and tracks Adam closely; subspace (SS) adapts inside the subspace and converges slower when the second moment is used. ProjFactor approximates OS with far lower memory.  
- **Budgeting and performance.** Define memory budget \(M=c·r\). Under the same \(M\), finer granularity generally performs better and can even reduce FLOPs and numeric error.   
- **Convergence.** With SGD, VLoRP attains the standard \(O(1/T)\) rate; with ProjFactor, a Lyapunov (Hamiltonian) argument shows monotone energy decrease toward a stationary point. 

---

## Highlights

- **Memory‑efficient:** stores projected gradients; total memory \(O(mn + 2nM + n + m)\). In experiments, VLoRP used ~**24.1GB** vs LoRA **28.8GB** (budget 256).  
- **Strong accuracy at fixed budget:** the finest‑grained setting \(c=256, r=1\) is consistently among the best under \(M=256\). 
- **Projection choices:** Gaussian (normal), Rademacher, or SVD bases all work similarly; normal is slightly better after convergence. 

---

## Install

```bash
# from source
pip install -e .

# optional: create a clean environment
# conda create -n vlorp python=3.10 -y && conda activate vlorp
# pip install -e .
```

> Requirements: Python ≥3.9, PyTorch ≥2.1, CUDA‑compatible GPUs recommended. (Adjust as needed for your setup.)

---

## Quick Start (Hugging Face Transformers)

> Below are **minimal templates**. Adjust file/flag names to your repo’s structure.

### A) CLI (if `scripts/finetune_hf.py` exists)
```bash
python scripts/finetune_hf.py \
  --model_id Qwen2-7B-Instruct \
  --dataset_path data/commonsense170k \
  --method vlorp \
  --budget 256 --c 256 --r 1 \
  --optimizer projfactor \
  --learning_rate 2e-4 --epochs 3 --batch_size 16 \
  --eval_tasks mmlu gsm8k arc-c arc-e boolq hellaswag obqa piqa siqa winogrande
```

### B) Python (library‑style)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# Illustrative API names; check the repo for exact imports.
from vlorp import apply_vlorp  # wraps gradient projection hooks
from vlorp.optim import ProjFactor

model_id = "Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure memory budget M=c*r with fine granularity
cfg = dict(c=256, r=1, budget=256)
apply_vlorp(model, **cfg)  # installs VLoRP projection hooks

optimizer = ProjFactor(model.parameters(), lr=2e-4)
# ... set up dataloaders, standard training loop ...
```

> Tip: Start with **fine granularity** (large `c`, small `r`) for a stronger baseline under the same `budget`. 

---

## Configuration

- `c` (**granularity**): number of vertical slices per gradient row (power‑of‑two recommended). Larger `c` ⇒ finer granularity.
- `r` (**rank**): subspace dimension per slice.
- `budget` (**M**): memory budget, product `c*r`; compare settings under the same `M`.
- `proj_type`: `normal` | `rademacher` | `svd` (all viable). 
- `scheme`: `os` (original‑space) or `ss` (subspace); `os` is recommended. 
- `optimizer`: `projfactor` (Adam‑like, memory‑saving) vs. `adam/adafactor`. 
- `tau` (optional): resampling gap for random projections (see paper appendix for ablations).

---

## Reproducing Paper‑Style Results

**Commonsense pre‑training → 8 eval tasks** (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC‑e/c, OBQA). Use LLaMA2‑7B bf16, activation checkpointing & grad accumulation enabled; set `M=256`, try `(c=256, r=1)`. 

**MMLU / GSM8K:** follow similar settings; OS + ProjFactor recommended for best memory/perf trade‑off. 

> Notes: exact hyperparameters, warmup steps, and projection refresh frequency are discussed in the paper’s appendix figures/tables.

---

## Why Fine Granularity?

Under a fixed budget `M`, finer granularity reduces FLOPs (by a factor proportional to `1/c`) and improves numeric stability—useful for bf16/fp16 and long accumulations.  

---

## Troubleshooting

- **Divergence with SS scheme:** switch to `os` or ProjFactor.
- **VRAM spikes with LoRP baselines:** ensure the *gradient tensor* is projected (not only optimizer states). VLoRP stores only projected gradients. 
- **bf16 stability:** prefer larger `c` (finer granularity) and moderate `r`.

---

## Roadmap

- Plug‑and‑play hooks for more HF architectures (Qwen, LLaMA‑3, etc.)
- Optimized CUDA kernels for projection‑back step
- More recipes & configs for public datasets

---

## Citation

If you use this repo, please cite the paper:

> *A Free Boost for Memory‑Efficient LLM Finetuning by Varying the Granularity of Gradient Projection (VLoRP), NeurIPS 2025 submission.*

```
@inproceedings{vlorp2025,
  title={A Free Boost for Memory-Efficient LLM Finetuning by Varying the Granularity of Gradient Projection},
  booktitle={NeurIPS},
  year={2025}
}
```

---

## License

Add your license here (e.g., Apache-2.0 or MIT).

---

## Acknowledgements

Inspired by LoRA/LoRP literature and memory‑efficient training works; see paper references for details.

