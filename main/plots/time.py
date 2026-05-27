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

C_TRAIN = '#1f77b4'  # Blue
C_FIT   = '#ff7f0e'  # Orange
C_EVAL  = '#2ca02c'  # Green

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

def infer_model_info_from_path(path_obj, json_data):
    m_type = json_data.get('type') or json_data.get('model_type')
    interp = json_data.get('interpolation')
    kernel = json_data.get('kernel')

    if m_type:
        m_type = m_type.lower()
        if m_type == 'qformer': m_type = 'q-former'
        return m_type, interp, kernel

    parts = [p.lower() for p in path_obj.parts]
    
    if 'ode-rnn' in parts: #remove - if not needed
        m_type = 'odernn'
    elif 'grud' in parts or 'gru-d' in parts:
        m_type = 'grud'
    elif 'log_ncde' in parts or 'log-ncde' in parts:
        m_type = 'log_ncde'
    elif 'baseline' in parts:
        m_type = 'baseline'
        if not interp:
            if 'cubic' in parts: interp = 'cubic'
            elif 'linear' in parts: interp = 'linear'
    elif 'qformer' in parts or 'q-former' in parts:
        m_type = 'q-former'
    elif 'conv' in parts:
        m_type = 'conv'
    elif 'gp' in parts:
        m_type = 'gp'
    elif 'kernel' in parts:
        m_type = 'kernel'
    else:
        m_type = 'unknown'

    if not kernel:
        if 'gp' in parts and m_type in ['q-former', 'conv', 'gp']:
            kernel = 'gp'
        elif 'gaussian' in parts or 'kernel' in parts:
            kernel = 'gaussian'
            
    return m_type, interp, kernel

def get_model_display_name(row):
    m_type = row.get('type')
    interp = row.get('interpolation')
    kernel = row.get('kernel', 'N/A')
    
    m_type = str(m_type) if m_type else 'unknown'
    
    if m_type == 'baseline':
        if interp == 'cubic': return 'Cubic'
        if interp == 'linear': return 'Linear'
        return f'Baseline ({interp})'
        
    if m_type == 'odernn': return 'ODE-RNN'
    if m_type == 'grud': return 'GRU-D'
    if m_type == 'log_ncde': return 'Log-NCDE'

    if m_type == 'kernel': return 'Gaussian'
    if m_type == 'gp': return 'GP'
    
    if m_type == 'q-former':
        if kernel == 'gp': return 'MV (GP)'
        return 'MV (Gauss)'
        
    if m_type == 'conv':
        if kernel == 'gp': return 'MVC (GP)'
        return 'MVC (Gauss)'
        
    return m_type

def load_data(base_dir, dataset_filter=None):
    data = []
    path = Path(base_dir)
    if not path.exists(): 
        print(f"Path {base_dir} does not exist.")
        return pd.DataFrame()

    json_files = list(path.rglob("*.json"))
    print(f"Found {len(json_files)} json files in {base_dir}")
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f: d = json.load(f)
            
            ds_name = d.get('dataset_name', d.get('dataset'))
            if not ds_name:
                for part in jf.parts:
                    if part in DATASETS:
                        ds_name = part
                        break
            
            if dataset_filter and ds_name != dataset_filter: continue
            if not ds_name: continue

            fit_t = d.get('trajectories_fit_time', d.get('train_fit_time', 0.0))
            train_t = d.get('training_time', d.get('train_pure_time', 0.0))
            
            eval_t = d.get('evaluation_time', 0.0)
            if eval_t == 0.0:
                eval_t = d.get('test_pure_time', 0.0) + d.get('test_fit_time', 0.0)

            if fit_t == 0.0 and train_t == 0.0:
                continue

            m_type, interp, kernel = infer_model_info_from_path(jf, d)

            row = {
                'dataset': ds_name,
                'fit': fit_t,
                'train': train_t,
                'eval': eval_t,
                'type': m_type,
                'interpolation': interp,
                'kernel': kernel,
            }
            
            display_name = get_model_display_name(row)
            if display_name.lower() == 'unknown':
                continue
                
            row['display_name'] = display_name
            data.append(row)
        except Exception: 
            pass
            
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_root", type=str, default="experiment_results_time_stats", 
                        help="Root dir containing experiment results")
    parser.add_argument("--output", type=str, default="pictures",
                        help="Output directory for plots")
    parser.add_argument("--filter", type=str, default=None,
                        help="Optional dataset name filter")
    args = parser.parse_args()

    print(f"Loading data from {args.plots_root}...")
    df = load_data(args.plots_root, args.filter)
    
    if df.empty:
        print("No valid time data found in directories.")
        return

    desired_order = [
        'Linear', 'Cubic', 'Log-NCDE', 'ODE-RNN', 'GRU-D',  
        'Gaussian', 'MV (Gauss)', 'MVC (Gauss)',
        'GP', 'MV (GP)', 'MVC (GP)'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    fig.suptitle("Computational Time Analysis", fontsize=20, fontweight='bold', y=0.98, x=0.52)
    
    axes[0, 0].set_ylabel("Pure Training Time (s)", fontweight='bold', fontsize=13)
    axes[1, 0].set_ylabel("Overhead Time (s)", fontweight='bold', fontsize=13)

    for col_idx, ds_name in enumerate(DATASETS):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        df_ds = df[df['dataset'] == ds_name]
        
        if df_ds.empty:
            ax_top.set_title(DATASET_ALIASES.get(ds_name, ds_name) + " (No Data)", fontsize=15, pad=10)
            ax_top.axis('off')
            ax_bot.axis('off')
            continue
            
        df_agg = df_ds.groupby('display_name')[['fit', 'train', 'eval']].mean()
        
        idx = [m for m in desired_order if m in df_agg.index]
        remaining = [m for m in df_agg.index if m not in idx]
        final_idx = idx + remaining
        
        df_agg = df_agg.reindex(final_idx)
        
        models = df_agg.index
        train_vals = df_agg['train'].values
        fit_vals = df_agg['fit'].values
        eval_vals = df_agg['eval'].values
        
        ax_top.bar(models, train_vals, color=C_TRAIN, alpha=0.85, edgecolor='black', width=0.7)
        ax_top.set_title(DATASET_ALIASES.get(ds_name, ds_name), fontsize=15, pad=10)
        ax_top.grid(axis='y', linestyle='--', alpha=0.4)

        ax_bot.bar(models, fit_vals, color=C_FIT, alpha=0.85, edgecolor='black', width=0.7)
        ax_bot.bar(models, eval_vals, bottom=fit_vals, color=C_EVAL, alpha=0.85, edgecolor='black', width=0.7)
        
        ax_bot.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.setp(ax_bot.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    patch_train = mpatches.Patch(color=C_TRAIN, label='Pure Training Time')
    patch_fit = mpatches.Patch(color=C_FIT, label='Trajectory Fit')
    patch_eval = mpatches.Patch(color=C_EVAL, label='Inference')
    
    fig.legend(handles=[patch_train, patch_fit, patch_eval], 
               loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=3, frameon=False, fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.90, wspace=0.15, hspace=0.1)
    
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "Combined_Time_Analysis.pdf")
    
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    
    print(f"Saved combined plot to: {out_path}")

if __name__ == "__main__":
    main()