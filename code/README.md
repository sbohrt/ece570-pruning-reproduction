# Tiny Reproduction: Magnitude-Based Neural Network Pruning
### ECE 57000 Course Project — Track 1: TinyReproductions

Reproduces the core accuracy-vs-sparsity claim from:
> Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). *Learning Both Weights and Connections for Efficient Neural Networks.* NeurIPS.

---

## Project Structure

```
code/
├── pruning_experiment.py   # Main experiment script (all code written by author)
├── README.md               # This file
└── results/                # Auto-created; contains generated figures and tables
    ├── accuracy_vs_sparsity.pdf
    ├── layer_sparsity_heatmap.pdf
    └── results_table.txt
paper/
├── main.tex                # ICLR 2026 format paper
├── references.bib          # Bibliography
├── iclr2026_conference.sty # ICLR 2026 style file
├── iclr2026_conference.bst # ICLR 2026 bibliography style
├── accuracy_vs_sparsity.pdf
└── layer_sparsity_heatmap.pdf
```

---

## Dependencies

- Python 3.8+
- PyTorch ≥ 1.12 (CPU, CUDA, or Apple Silicon MPS)
- torchvision
- matplotlib
- numpy

Install via:
```bash
pip install torch torchvision matplotlib numpy
```

---

## How to Run

From the `code/` directory:

```bash
python pruning_experiment.py
```

MNIST data is downloaded automatically to `./data/` on first run (~11 MB).

**Expected runtime:**
- Apple Silicon (MPS): ~8–12 minutes
- CPU: ~20–30 minutes
- CUDA GPU: ~5–8 minutes

The script automatically selects CUDA → MPS → CPU in that order.

Results are saved to `./results/`.

---

## Dataset

MNIST is downloaded automatically via `torchvision.datasets.MNIST`.
No manual download is needed.

---

## Expected Output

```
Device: mps   (or cuda / cpu)

[1/5] Training baseline model (10 epochs)...
  [baseline] epoch  1 — test acc 0.9472
  ...
  [baseline] epoch 10 — test acc 0.9836
Baseline test accuracy: 0.9836

[2/5] One-shot pruning sweep...
  target= 0%  actual= 0.0%  acc=0.9836
  target=30%  actual=30.0%  acc=0.9823
  ...
  target=95%  actual=95.0%  acc=0.5667

[3/5] Iterative pruning sweep...
  ...
  target=95%  → final acc=0.9651  actual sparsity=94.9%

[4/5] Global vs. layer-wise comparison at 90%...
[5/5] Layer compression analysis...

Saved: results/accuracy_vs_sparsity.pdf
Saved: results/layer_sparsity_heatmap.pdf
Saved: results/results_table.txt
```

---

## Code Authorship

All code in `pruning_experiment.py` was written by the author for this project.

- **MLP architecture** (lines 45–56) and **global magnitude pruning** (lines 103–109):
  first developed in Checkpoint 1.
- **Fine-tuning with gradient masking** (lines 120–136) and **iterative pruning loop**
  (lines 139–149): developed in Checkpoint 2.
- **Layer-wise pruning** (lines 112–117), **global vs. layer-wise comparison**
  (lines 199–228), **layer compression analysis** (lines 229–242), and **plotting
  utilities** (lines 244–310): new for the final submission.
- **MPS device support** and **SSL certificate patch** for macOS: added during
  the final submission phase.

No code was copied from external repositories. LLM assistance (Claude Sonnet 4.6) was
used for code structure suggestions and LaTeX formatting; all implementation logic,
experimental design, and analysis are the author's own.

