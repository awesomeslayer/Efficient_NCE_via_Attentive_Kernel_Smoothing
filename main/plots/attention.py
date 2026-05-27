import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import pickle
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

DATASETS = ["CharacterTrajectories", "SpokenArabicDigits", "UWaveGestureLibrary"]
DATASET_ALIASES = {
    "CharacterTrajectories": "Character Trajectories",
    "SpokenArabicDigits": "Spoken Arabic Digits",
    "UWaveGestureLibrary": "UWave Gesture"
}
MODELS = ["qformer", "conv"]
MODEL_ALIASES = {
    "qformer": "MV-CDE (Ours)",
    "conv": "MVC-CDE (Ours)"
}

TARGET_REGULARIZATION = "reg-on" 
CHERRY_PICK_INDEX = 4 

def find_experiment_data(ds_path, model_type):
    attn_dir = ds_path / "attention_maps"
    if not attn_dir.exists(): return None
    candidates = [p for p in attn_dir.glob("*.pkl") if model_type in p.name.lower() and 'gp' in p.name.lower() and TARGET_REGULARIZATION in p.name.lower()]
    if not candidates: return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

def load_data(path, sample_idx=0):
    try:
        with open(path, 'rb') as f: 
            d = pickle.load(f)
        attns, Xs = d.get('fixed_attns', None), d.get('fixed_Xs', None)
        if attns is None or Xs is None: 
            return None, None
        idx = min(sample_idx, len(attns) - 1)
        return Xs[idx].numpy() if hasattr(Xs[idx], 'numpy') else Xs[idx], attns[idx].numpy() if hasattr(attns[idx], 'numpy') else attns[idx]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_root", type=str, default="experiment_attention")
    parser.add_argument("--output", type=str, default="pictures")
    args = parser.parse_args()
    root, out_dir = Path(args.plots_root), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating Overlapped Grid (Cherry-pick index: {CHERRY_PICK_INDEX})...")

    fig = plt.figure(figsize=(18, 9))
    
    fig.suptitle(f"Multi-Model Attention Comparison on Single Trajectory", fontsize=22, fontweight='bold', y=0.96)
    
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.8, 1.5, 1.5], hspace=0.12, wspace=0.15)

    legend_handles, legend_labels = [], []
    
    for col, ds_name in enumerate(DATASETS):
        ds_path = root / ds_name
        
        p1 = find_experiment_data(ds_path, MODELS[0])
        p2 = find_experiment_data(ds_path, MODELS[1])
        X1, attn1 = load_data(p1, CHERRY_PICK_INDEX) if p1 else (None, None)
        X2, attn2 = load_data(p2, CHERRY_PICK_INDEX) if p2 else (None, None)
        
        X = X1 if X1 is not None else X2

        ax_traj = plt.subplot(gs[0, col])
        if X is not None:
            seq_len = X.shape[0]
            time_steps = np.arange(seq_len)
            ax_traj.plot(time_steps, X[:, 0], color='#2c3e50', linewidth=2, marker='o', markersize=3)
            ax_traj.set_xlim(-0.5, seq_len - 0.5)
            ax_traj.yaxis.set_major_locator(plt.MaxNLocator(3))
        else:
            ax_traj.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
        
        ax_traj.grid(True, linestyle='--', alpha=0.5)
        plt.setp(ax_traj.get_xticklabels(), visible=False)
        ax_traj.set_title(DATASET_ALIASES[ds_name], fontsize=16, pad=12)
        if col == 0: 
            ax_traj.set_ylabel("Original\nSignal", fontweight='bold')

        ax_m1 = plt.subplot(gs[1, col], sharex=ax_traj)
        if attn1 is not None:
            n_heads = attn1.shape[0]
            colors = sns.color_palette("Set1", n_colors=n_heads)
            for h in range(n_heads):
                ax_m1.fill_between(time_steps, attn1[h, :], step='mid', alpha=0.3, color=colors[h])
                line, = ax_m1.plot(time_steps, attn1[h, :], drawstyle='steps-mid', color=colors[h], linewidth=2)
                if col == 0 and len(legend_handles) < n_heads:
                    legend_handles.append(line)
                    legend_labels.append(f"Head {h+1}")
            ax_m1.set_ylim(0, np.max(attn1) * 1.15)
        
        ax_m1.grid(True, linestyle='--', alpha=0.4)
        plt.setp(ax_m1.get_xticklabels(), visible=False)
        if col == 0: 
            ax_m1.set_ylabel(f"{MODEL_ALIASES[MODELS[0]]}\nAttention", fontweight='bold')

        ax_m2 = plt.subplot(gs[2, col], sharex=ax_traj)
        if attn2 is not None:
            n_heads = attn2.shape[0]
            colors = sns.color_palette("Set1", n_colors=n_heads)
            for h in range(n_heads):
                ax_m2.fill_between(time_steps, attn2[h, :], step='mid', alpha=0.3, color=colors[h])
                ax_m2.plot(time_steps, attn2[h, :], drawstyle='steps-mid', color=colors[h], linewidth=2)
            ax_m2.set_ylim(0, np.max(attn2) * 1.15)
        
        ax_m2.grid(True, linestyle='--', alpha=0.4)
        if col == 0: 
            ax_m2.set_ylabel(f"{MODEL_ALIASES[MODELS[1]]}\nAttention", fontweight='bold')

    fig.text(0.53, 0.10, "Time Step", ha='center', va='center', fontweight='bold', fontsize=16)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                   ncol=len(legend_labels), bbox_to_anchor=(0.53, 0.01), 
                   fontsize=14, frameon=True)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.16)
    
    out_file = out_dir / f"Attention_Overlapped_Shared_Cherry{CHERRY_PICK_INDEX}.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_file.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"Saved Overlapped grid to: {out_file}")

if __name__ == "__main__":
    main()