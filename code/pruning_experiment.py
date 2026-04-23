"""
Tiny Reproduction: Learning Both Weights and Connections (Han et al., 2015)
Compares one-shot vs. iterative magnitude pruning on MNIST with an MLP.
"""

import os
import ssl
import copy
import random
import numpy as np
import torch

# macOS ships with an outdated SSL certificate bundle; this lets torchvision
# download MNIST over HTTPS without a certificate verification failure.
ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

PIN_MEMORY = DEVICE.type == "cuda"  # pin_memory only works reliably with CUDA

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders(batch_size=128):
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)
    return train_loader, test_loader


# ── Training helpers ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += yb.size(0)
    return correct / total


def train_baseline(epochs=10):
    train_loader, test_loader = get_loaders()
    model = MLP().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    for e in range(1, epochs + 1):
        train_epoch(model, train_loader, opt, crit)
        acc = evaluate(model, test_loader)
        print(f"  [baseline] epoch {e:2d} — test acc {acc:.4f}")
    return model, test_loader, train_loader


# ── Pruning ────────────────────────────────────────────────────────────────────
def global_magnitude_prune(model, sparsity):
    """Zero out the lowest-|w| fraction of all weights globally."""
    all_weights = torch.cat([p.view(-1).abs() for p in model.parameters()])
    threshold   = torch.quantile(all_weights, sparsity)
    with torch.no_grad():
        for p in model.parameters():
            p *= (p.abs() >= threshold).float()


def layerwise_magnitude_prune(model, sparsity):
    """Zero out the lowest-|w| fraction within each layer independently."""
    with torch.no_grad():
        for p in model.parameters():
            threshold = torch.quantile(p.abs().view(-1), sparsity)
            p        *= (p.abs() >= threshold).float()


def fine_tune(model, loader, test_loader, epochs=3, lr=1e-3):
    """Fine-tune while holding pruned weights at zero (gradient masking)."""
    masks     = {id(p): (p != 0).float() for p in model.parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            with torch.no_grad():
                for p in model.parameters():
                    if id(p) in masks:
                        p.grad *= masks[id(p)]   # keep pruned weights dead
            optimizer.step()
    return evaluate(model, test_loader)


def iterative_prune(model, train_loader, test_loader, target_sparsity, rounds, ft_epochs=3):
    """Incrementally prune by target_sparsity/rounds per round, fine-tuning each time."""
    history = []
    sparsity_step = target_sparsity / rounds
    for r in range(1, rounds + 1):
        current = sparsity_step * r
        global_magnitude_prune(model, current)
        acc = fine_tune(model, train_loader, test_loader, epochs=ft_epochs)
        history.append((current, acc))
        print(f"    round {r:2d}: sparsity={current:.0%}  acc={acc:.4f}")
    return history


def measure_sparsity(model):
    """Return dict of per-layer and global zero-weight fractions."""
    result = {}
    total_zeros = total_params = 0
    for name, p in model.named_parameters():
        zeros  = (p == 0).sum().item()
        params = p.numel()
        result[name] = zeros / params
        total_zeros  += zeros
        total_params += params
    result["global"] = total_zeros / total_params
    return result


# ── Experiments ────────────────────────────────────────────────────────────────
SPARSITIES = [0.0, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]


def run_oneshot(baseline_model, test_loader):
    print("\n=== One-shot pruning sweep ===")
    results = []
    for s in SPARSITIES:
        m = copy.deepcopy(baseline_model)
        if s > 0:
            global_magnitude_prune(m, s)
        acc = evaluate(m, test_loader)
        sp  = measure_sparsity(m)["global"]
        results.append((s, acc, sp))
        print(f"  target={s:.0%}  actual={sp:.1%}  acc={acc:.4f}")
    return results


def run_iterative(baseline_model, train_loader, test_loader):
    print("\n=== Iterative pruning sweep ===")
    results = [(0.0, evaluate(copy.deepcopy(baseline_model), test_loader), 0.0)]
    for s in SPARSITIES[1:]:
        print(f"  target sparsity {s:.0%}")
        m = copy.deepcopy(baseline_model)
        rounds = max(3, int(s / 0.10))
        hist   = iterative_prune(m, train_loader, test_loader, s, rounds=rounds)
        acc    = hist[-1][1]
        sp     = measure_sparsity(m)["global"]
        results.append((s, acc, sp))
        print(f"  → final acc={acc:.4f}  actual sparsity={sp:.1%}")
    return results


def run_layerwise(baseline_model, test_loader, train_loader):
    """Compare global vs. layer-wise pruning at a fixed target sparsity."""
    print("\n=== Layer-wise vs. global pruning at 90% sparsity ===")
    target = 0.90

    m_global = copy.deepcopy(baseline_model)
    global_magnitude_prune(m_global, target)
    acc_global = evaluate(m_global, test_loader)
    sp_global  = measure_sparsity(m_global)
    print(f"  Global   acc={acc_global:.4f}  actual={sp_global['global']:.1%}")

    m_layer = copy.deepcopy(baseline_model)
    layerwise_magnitude_prune(m_layer, target)
    acc_layer = evaluate(m_layer, test_loader)
    sp_layer  = measure_sparsity(m_layer)
    print(f"  Layerwise acc={acc_layer:.4f}  actual={sp_layer['global']:.1%}")

    m_iter = copy.deepcopy(baseline_model)
    iterative_prune(m_iter, train_loader, test_loader, target, rounds=9)
    acc_iter = evaluate(m_iter, test_loader)
    sp_iter  = measure_sparsity(m_iter)
    print(f"  Iterative acc={acc_iter:.4f}  actual={sp_iter['global']:.1%}")

    return {
        "global":    (acc_global, sp_global),
        "layerwise": (acc_layer,  sp_layer),
        "iterative": (acc_iter,   sp_iter),
    }


def run_layer_compression_analysis(baseline_model, train_loader, test_loader):
    """Reproduce Han et al. Fig. 3: per-layer sparsity after global pruning."""
    print("\n=== Layer compression analysis ===")
    model = copy.deepcopy(baseline_model)
    # Use iterative pruning to reach 90% so we measure the "trained" mask
    iterative_prune(model, train_loader, test_loader, 0.90, rounds=9)
    sp = measure_sparsity(model)
    for k, v in sp.items():
        if k != "global":
            print(f"  {k}: {v:.1%} zeros")
    print(f"  global: {sp['global']:.1%} zeros")
    return sp


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_accuracy_vs_sparsity(oneshot, iterative):
    s_os,  a_os  = zip(*[(r[0], r[1]) for r in oneshot])
    s_it,  a_it  = zip(*[(r[0], r[1]) for r in iterative])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([x * 100 for x in s_os], [x * 100 for x in a_os],
            "o-", color="#e05c5c", label="One-Shot Pruning",      linewidth=1.8)
    ax.plot([x * 100 for x in s_it], [x * 100 for x in a_it],
            "s-", color="#4caf7d", label="Iterative + Fine-Tune", linewidth=1.8)
    ax.axhline(y=a_os[0] * 100, color="gray", linestyle="--", linewidth=1, label="Dense Baseline")

    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Sparsity: One-Shot vs. Iterative Pruning", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(-2, 97)
    ax.set_ylim(82, 99.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_vs_sparsity.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_layer_sparsity_heatmap(sparsity_dict):
    layer_names = [k for k in sparsity_dict if k != "global"]
    values      = [sparsity_dict[k] * 100 for k in layer_names]
    nice_names  = ["fc1\n(784→300)", "fc2\n(300→100)", "fc3\n(100→10)",
                   "fc1 bias", "fc2 bias", "fc3 bias"][:len(layer_names)]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    bars = ax.barh(nice_names, values, color="#4a90d9", edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percentage of Weights Zeroed (%)", fontsize=11)
    ax.set_title("Per-Layer Sparsity After Iterative Pruning (90% Global Target)", fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "layer_sparsity_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_results_table(oneshot, iterative, lw_results):
    path = os.path.join(RESULTS_DIR, "results_table.txt")
    with open(path, "w") as f:
        f.write("Sparsity | One-Shot Acc | Iterative Acc\n")
        f.write("-" * 44 + "\n")
        for (s_os, a_os, _), (s_it, a_it, _) in zip(oneshot, iterative):
            f.write(f"  {s_os:.0%}    |    {a_os*100:.2f}%    |    {a_it*100:.2f}%\n")
        f.write("\nLayerwise comparison at 90% sparsity:\n")
        for method, (acc, sp) in lw_results.items():
            f.write(f"  {method:12s}: acc={acc*100:.2f}%  global_sparsity={sp['global']*100:.1f}%\n")
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    print("\n[1/5] Training baseline model (10 epochs)...")
    baseline_model, test_loader, train_loader = train_baseline(epochs=10)
    baseline_acc = evaluate(baseline_model, test_loader)
    print(f"\nBaseline test accuracy: {baseline_acc:.4f}")

    print("\n[2/5] One-shot pruning sweep...")
    oneshot_results = run_oneshot(baseline_model, test_loader)

    print("\n[3/5] Iterative pruning sweep...")
    iterative_results = run_iterative(baseline_model, train_loader, test_loader)

    print("\n[4/5] Global vs. layer-wise comparison at 90%...")
    lw_results = run_layerwise(baseline_model, test_loader, train_loader)

    print("\n[5/5] Layer compression analysis...")
    layer_sparsity = run_layer_compression_analysis(baseline_model, train_loader, test_loader)

    print("\n--- Generating figures ---")
    plot_accuracy_vs_sparsity(oneshot_results, iterative_results)
    plot_layer_sparsity_heatmap(layer_sparsity)
    save_results_table(oneshot_results, iterative_results, lw_results)

    print("\nAll done. Results saved to ./results/")
