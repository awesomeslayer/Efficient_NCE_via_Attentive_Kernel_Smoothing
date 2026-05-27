import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import argparse
from pathlib import Path
import numpy as np
import ast

plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
})

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

DATASETS = ["CharacterTrajectories", "SpokenArabicDigits", "UWaveGestureLibrary"]
DATASET_ALIASES = {
    "CharacterTrajectories": "Character Trajectories",
    "SpokenArabicDigits": "Spoken Arabic Digits",
    "UWaveGestureLibrary": "UWave Gesture"
}

KERNELS = ['gaussian', 'gp']
KERNEL_ALIASES = {'gaussian': 'Gaussian', 'gp': 'GP'}

def parse_list_str(s):
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, list) else []
    except:
        return []

def load_and_prep_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except:
        return pd.DataFrame()

    if 'model_type' in df.columns:
        df['model_type'] = df['model_type'].replace({'qformer': 'q-former'})

    df = df[df['model_type'].isin(['q-former', 'conv'])].copy()
    
    def extract_head_info(row):
        raw_str = str(row.get('bandwidths_str', '[]'))
        lst = parse_list_str(raw_str) if raw_str not in ['nan', 'N/A', 'None'] else []
        if not lst: return 0, 0.0, False
        first_val = float(lst[0])
        is_same = all(abs(float(x) - first_val) < 1e-9 for x in lst)
        return len(lst), first_val, is_same

    info = df.apply(extract_head_info, axis=1, result_type='expand')
    df['num_heads'], df['bw_value'], df['is_same_mode'] = info[0], info[1], info[2]
    df = df[df['is_same_mode'] & (df['num_heads'] > 0)]

    if 'test_acc' in df.columns:
        df['test_acc'] = pd.to_numeric(df['test_acc'], errors='coerce')
        df['error_rate'] = 1.0 - (df['test_acc'] / 100.0)
    
    return df

def get_aggregated(df):
    group_cols = ['dataset', 'model_type', 'kernel_func', 'num_heads', 'bw_value']
    agg = df.groupby(group_cols)[['error_rate']].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    return agg

def filter_complete_series(df):
    required_heads = set(range(1, 2)) 
    config_cols = ['dataset', 'model_type', 'kernel_func', 'bw_value']
    valid_indices = []
    for _, group in df.groupby(config_cols):
        present_heads = set(group['num_heads'].unique())
        valid_indices.extend(group.index.tolist())
    return df.loc[valid_indices].copy()

def plot_final_grid(df, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=False)
    fig.suptitle("Error Rate vs Number of Heads", fontsize=18, fontweight='bold', x=0.54, y=0.98)

    for col_idx, ds_name in enumerate(DATASETS):
        df_ds = df[df['dataset'] == ds_name]
        unique_bws = sorted(df_ds['bw_value'].unique())
        
        ds_handles = []

        for row_idx, kernel in enumerate(KERNELS):
            ax = axes[row_idx, col_idx]
            df_k = df_ds[df_ds['kernel_func'] == kernel]
            if kernel == 'gaussian':
                 if df_k.empty:
                     df_k = df_ds[df_ds['kernel_func'].isin(['rbf', 'gaussian'])]

            for i, bw in enumerate(unique_bws):
                color = COLORS[i % len(COLORS)]
                
                if row_idx == 0:
                    ds_handles.append(mlines.Line2D([], [], color=color, label=f"h={bw}"))

                sub_mv = df_k[(df_k['model_type'] == 'q-former') & (df_k['bw_value'] == bw)].sort_values('num_heads')
                sub_mv = sub_mv[sub_mv['num_heads'].between(1, 8)]
                if not sub_mv.empty:
                    ax.errorbar(sub_mv['num_heads'], sub_mv['error_rate_mean'], yerr=sub_mv['error_rate_std'],
                                marker='o', linestyle='-', color=color, alpha=0.5, capsize=3)

                sub_mvc = df_k[(df_k['model_type'] == 'conv') & (df_k['bw_value'] == bw)].sort_values('num_heads')
                sub_mvc = sub_mvc[sub_mvc['num_heads'].between(1, 8)]
                if not sub_mvc.empty:
                    ax.errorbar(sub_mvc['num_heads'], sub_mvc['error_rate_mean'], yerr=sub_mvc['error_rate_std'],
                                marker='*', linestyle='--', color=color, alpha=0.5, capsize=3)

            ax.grid(True, ls='--', alpha=0.4)
            ax.xaxis.set_major_locator(ticker.FixedLocator(range(1, 9)))

        if ds_handles:
            axes[0, col_idx].legend(handles=ds_handles, loc='upper center', 
                                    bbox_to_anchor=(0.5, 1.02), 
                                    ncol=min(4, len(ds_handles)), frameon=False, 
                                    columnspacing=0.8, handletextpad=0.2)
        
        axes[0, col_idx].set_title(DATASET_ALIASES[ds_name], pad=35, fontsize=14)
        axes[0, 0].set_ylabel("Gaussian", fontsize=13, labelpad=15)
        axes[1, 0].set_ylabel("GP", fontsize=13, labelpad=15)

    fig.text(0.04, 0.5, "Error Rate", va='center', rotation='vertical', fontsize=15, fontweight='bold')
    fig.text(0.54, 0.07, "Number of Heads", ha='center', fontsize=15, fontweight='bold')

    line_mv = mlines.Line2D([], [], color='gray', marker='o', linestyle='-', label='MV-CDE')
    line_mvc = mlines.Line2D([], [], color='gray', marker='*', linestyle='--', label='MVC-CDE')
    fig.legend(handles=[line_mv, line_mvc], loc='lower center', bbox_to_anchor=(0.54, 0.01), 
               ncol=2, frameon=False, fontsize=13)

    plt.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.15, wspace=0.2, hspace=0.1)
    
    out_file = output_dir / "Heads_Final_Grid.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    png_path = out_file.with_suffix('.png') 
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="experiment_results_add_moreheads/summary_results.csv")
    parser.add_argument("--out_dir", type=str, default="pictures")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_and_prep_data(args.input)
    if df.empty: return

    df_agg = get_aggregated(df)
    df_filtered = filter_complete_series(df_agg)
    
    if df_filtered.empty:
        print("No complete 1-8 series found.")
        return

    plot_final_grid(df_filtered, out_dir)

if __name__ == "__main__":
    main()