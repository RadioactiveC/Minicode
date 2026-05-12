"""Generate training and evaluation plots for GitHub README.

Outputs:
  eval/train_loss.png      - Training loss curve
  eval/eval_comparison.png - Base vs LoRA accuracy comparison
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EVAL_DIR = os.path.join(PROJECT_DIR, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

# ── 1. Training Loss Curve ──

loss_data = [1.0742, 0.9072, 0.8360, 0.8096, 0.8015]
epochs = list(range(1, len(loss_data) + 1))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, loss_data, 'o-', color='#2563eb', linewidth=2.5, markersize=8, label='Training Loss')
ax.fill_between(epochs, loss_data, alpha=0.1, color='#2563eb')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Average Loss', fontsize=13)
ax.set_title('MinCode LoRA Training Loss', fontsize=15, fontweight='bold')
ax.set_xticks(epochs)
ax.set_ylim(0.6, 1.2)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

# Annotate each point
for x, y in zip(epochs, loss_data):
    ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=10, color='#1e40af')

plt.tight_layout()
loss_path = os.path.join(EVAL_DIR, "train_loss.png")
fig.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {loss_path}")

# ── 2. Evaluation Comparison ──

categories = ['list_files', 'read_file', 'write_file', 'no_tool', 'edge', 'TOTAL']

# Load eval results
base_data = {"list_files": 100, "read_file": 0, "write_file": 0, "no_tool": 100, "edge": 0, "TOTAL": 40.0}
lora_data = {"list_files": 100, "read_file": 100, "write_file": 33, "no_tool": 67, "edge": 67, "TOTAL": 73.3}

# Try loading from files
for fname, data_dict in [("eval_full_sft.json", base_data), ("eval_mincode_sft.json", lora_data)]:
    fpath = os.path.join(EVAL_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        for cat, stats in d["summary"]["by_category"].items():
            data_dict[cat] = stats["accuracy"]
        data_dict["TOTAL"] = d["summary"]["accuracy"]

base_vals = [base_data[c] for c in categories]
lora_vals = [lora_data[c] for c in categories]

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(categories))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], base_vals, width, label='Base (full_sft)',
               color='#94a3b8', edgecolor='white', linewidth=0.5)
bars2 = ax.bar([i + width/2 for i in x], lora_vals, width, label='LoRA (mincode_sft)',
               color='#2563eb', edgecolor='white', linewidth=0.5)

ax.set_xlabel('Category', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Tool-Call Accuracy: Base vs LoRA', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)

# Annotate bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.0f}%',
            ha='center', va='bottom', fontsize=9, color='#475569')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.0f}%',
            ha='center', va='bottom', fontsize=9, color='#1e40af', fontweight='bold')

plt.tight_layout()
eval_path = os.path.join(EVAL_DIR, "eval_comparison.png")
fig.savefig(eval_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {eval_path}")
