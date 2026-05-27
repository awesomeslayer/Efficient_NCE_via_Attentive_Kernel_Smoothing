import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging

DATASETS = ["CharacterTrajectories", "SpokenArabicDigits", "UWaveGestureLibrary"]
DATASET_ALIASES = {
    "CharacterTrajectories": "Character Trajectories",
    "SpokenArabicDigits": "Spoken Arabic Digits",
    "UWaveGestureLibrary": "UWave Gesture"
}

C_FIT   = '#ff7f0e'  
C_EVAL  = '#2ca02c'  

plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18,
})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_model_display_name(jf_path):
    """Парсит имя модели на основе имени файла/пути"""
    name = jf_path.name.lower()
    
    if 'baseline_cubic' in name: return 'Cubic'
    if 'baseline_linear' in name: return 'Linear'
    if 'grud' in name: return 'GRU-D'
    if 'logncde' in name: return 'Log-NCDE'
    
    if name.startswith('kernel-'): return 'Gaussian'
    if name.startswith('gp_'): return 'GP'
    
    if 'qformer' in name:
        if 'gp' in name: return 'MV (GP)'
        return 'MV (Gauss)'
        
    if 'conv' in name:
        if 'gp' in name: return 'MVC (GP)'
        return 'MVC (Gauss)'
        
    return 'Unknown'

def load_data(base_dir, dataset_filter=None):
    data = []
    path = Path(base_dir)
    if not path.exists(): 
        logger.error(f"Path {base_dir} does not exist.")
        return pd.DataFrame()

    json_files = list(path.rglob("*.json"))
    logger.info(f"Found {len(json_files)} json files in {base_dir}")
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f: 
                d = json.load(f)
            
            ds_name = d.get('dataset_name', d.get('dataset'))
            if not ds_name:
                for part in jf.parts:
                    if part in DATASETS:
                        ds_name = part
                        break
            
            if dataset_filter and ds_name != dataset_filter: continue
            if not ds_name: continue

            fit_t = d.get('test_fit_time', 0.0)
            eval_t = d.get('test_pure_time', 0.0)

            if fit_t == 0.0 and eval_t == 0.0:
                continue

            display_name = get_model_display_name(jf)
            if display_name == 'Unknown':
                continue

            data.append({
                'dataset': ds_name,
                'fit': fit_t,
                'eval': eval_t,
                'display_name': display_name
            })
        except Exception as e:
            logger.warning(f"Error reading {jf}: {e}")
            
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_root", type=str, default="experiment_times", 
                        help="Root dir containing experiment results")
    parser.add_argument("--output", type=str, default="plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.plots_root}...")
    df = load_data(args.plots_root)
    
    if df.empty:
        logger.error("No valid time data found in directories.")
        return

    desired_order = [
        'Linear', 'Cubic', 'Log-NCDE',  'GRU-D',  
        'Gaussian', 'MV (Gauss)', 'MVC (Gauss)',
        'GP', 'MV (GP)', 'MVC (GP)'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    fig.suptitle("Inference Time Analysis (Pre-computation vs Forward Pass)", 
                 fontsize=18, fontweight='bold', y=1.05)
    
    axes[0].set_ylabel("Time (seconds)", fontweight='bold', fontsize=13)

    for col_idx, ds_name in enumerate(DATASETS):
        ax = axes[col_idx]
        df_ds = df[df['dataset'] == ds_name]
        
        if df_ds.empty:
            ax.set_title(DATASET_ALIASES.get(ds_name, ds_name) + "\n(No Data)", fontsize=15, pad=10)
            ax.axis('off')
            continue
            
        df_agg = df_ds.groupby('display_name')[['fit', 'eval']].mean()
        
        idx = [m for m in desired_order if m in df_agg.index]
        remaining = [m for m in df_agg.index if m not in idx]
        final_idx = idx + remaining
        df_agg = df_agg.reindex(final_idx).fillna(0)
        
        models = df_agg.index
        fit_vals = df_agg['fit'].values
        eval_vals = df_agg['eval'].values
        
        ax.bar(models, fit_vals, color=C_FIT, alpha=0.85, edgecolor='black', width=0.6)
        ax.bar(models, eval_vals, bottom=fit_vals, color=C_EVAL, alpha=0.85, edgecolor='black', width=0.6)
        
        ax.set_title(DATASET_ALIASES.get(ds_name, ds_name), fontsize=15, pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True) 
        
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    patch_fit = mpatches.Patch(color=C_FIT, label='Fit Time (Interpolation Pre-calc)')
    patch_eval = mpatches.Patch(color=C_EVAL, label='Inference Time (Forward Pass)')
    
    fig.legend(handles=[patch_fit, patch_eval], 
               loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=2, frameon=False, fontsize=13)

    plt.tight_layout()
    
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "Inference_Time_Tradeoff.pdf")
    
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()