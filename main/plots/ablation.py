import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import argparse
from pathlib import Path
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 6, 
    'figure.autolayout': False,
})

# Colors
STYLE = {
    'baseline-linear': {'color': 'tab:cyan',   'marker': 'd', 'label': 'Linear'},
    'baseline-cubic':  {'color': 'tab:olive',  'marker': 'X', 'label': 'Cubic'},
    'odernn':          {'color': 'tab:pink',   'marker': 'h', 'label': 'ODE-RNN'}, 
    'grud':            {'color': 'black',      'marker': '^', 'label': 'GRU-D'},   
    'kernel':          {'color': 'tab:blue',   'marker': 'o', 'label': 'Gaussian CDE'},
    'gp':              {'color': 'tab:orange', 'marker': 'p', 'label': 'GP CDE'},
    'q-former':        {'color': 'tab:green',  'marker': 's', 'label': 'MV-CDE (Gaussian)'},
    'conv':            {'color': 'tab:purple', 'marker': '*', 'label': 'MVC-CDE (Gaussian)'},
    'q-former-gp':     {'color': 'tab:red',    'marker': 'v', 'label': 'MV-CDE (GP)'},
    'conv-gp':         {'color': 'tab:brown',  'marker': 'D', 'label': 'MVC-CDE (GP)'},
}

def load_data(csv_path):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    
    if 'train_all_time' in df.columns:
        df['train_time_sec'] = pd.to_numeric(df['train_all_time'], errors='coerce')
    
    if 'model_type' in df.columns:
        df['model_type'] = df['model_type'].replace({'qformer': 'q-former'})

    for c in ['test_acc', 'train_time_sec', 'seed']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    if 'test_acc' in df.columns:
        df['test_error'] = 1.0 - (df['test_acc'] / 100.0)
        df['test_error'] = df['test_error'].clip(lower=1e-6)
    
    fill_cols = ['dataset', 'model_type', 'kernel_func', 'interpolation', 
                 'bw_mode', 'bandwidths_str', 'length_scale', 'tolerance',
                 'time_scaling', 'aggregation']
    for c in fill_cols:
        if c in df.columns: df[c] = df[c].fillna('N/A').astype(str)
        else: df[c] = 'N/A'
            
    return df

def get_aggregated(df):
    if df.empty: return df
    group_cols = ['dataset', 'model_type', 'kernel_func', 'interpolation', 
                  'bw_mode', 'bandwidths_str', 'length_scale', 'tolerance',
                  'time_scaling', 'aggregation']
    valid = [c for c in group_cols if c in df.columns]
    agg = df.groupby(valid, dropna=False)[['test_error', 'train_time_sec']].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(c).strip() if c[1] else c[0] for c in agg.columns.values]
    return agg

def plot_series(ax, df_sub, style_key, emphasize=False):
    if df_sub.empty: return
    s = STYLE.get(style_key, {'color': 'k', 'marker': '.', 'label': style_key})
    
    x = df_sub['train_time_sec_mean']
    y = df_sub['test_error_mean']
    x_err = df_sub['train_time_sec_std'].fillna(0)
    y_err = df_sub['test_error_std'].fillna(0)
    
    if emphasize:
        alpha_points = 1.0
        alpha_bars = 1.0
        size = 130 
        zorder = 10
        edge_c = 'black'
        lw = 1.5
    else:
        alpha_points = 0.6 
        alpha_bars = 0.3
        size = 60
        zorder = 5
        edge_c = 'white'
        lw = 0.5

    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', 
                ecolor=s['color'], alpha=alpha_bars, elinewidth=1, capsize=0, zorder=zorder-1)
    
    ax.scatter(x, y, c=s['color'], marker=s['marker'], s=size, 
               edgecolors=edge_c, linewidth=lw, alpha=alpha_points, zorder=zorder)

def plot_baselines(ax, df_ds):
    plot_series(ax, df_ds[(df_ds['model_type'] == 'baseline') & (df_ds['interpolation'] == 'linear')], 'baseline-linear')
    plot_series(ax, df_ds[(df_ds['model_type'] == 'baseline') & (df_ds['interpolation'] == 'cubic')], 'baseline-cubic')
    plot_series(ax, df_ds[df_ds['model_type'] == 'odernn'], 'odernn')
    plot_series(ax, df_ds[df_ds['model_type'] == 'grud'], 'grud')

def get_best(df_sub):
    if df_sub.empty: return df_sub
    return df_sub.loc[[df_sub['test_error_mean'].idxmin()]]

def configure_axis(ax, title):
    ax.set_title(title, pad=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.grid(True, which="major", ls="-", alpha=0.3, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.15, color='gray')
    
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

def create_global_legend(fig, mode_keys):
    handles = []
    labels = []
    
    base_keys = ['baseline-linear', 'baseline-cubic', 'odernn', 'grud']
    all_keys = base_keys + mode_keys
    
    seen = set()
    unique_keys = [x for x in all_keys if not (x in seen or seen.add(x))]

    for k in unique_keys:
        s = STYLE[k]
        h = mlines.Line2D([], [], color=s['color'], marker=s['marker'], 
                          linestyle='None', markersize=10, label=s['label'])
        handles.append(h)
        labels.append(s['label'])
        
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
               ncol=5, frameon=False, columnspacing=1.0)


def plot_combined_figure(df_agg, mode, out_path):
    datasets = sorted(df_agg['dataset'].unique())
    n_ds = len(datasets)
    if n_ds == 0: return

    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), sharex=False, sharey=False)
    if n_ds == 1: axes = [axes]
    
    legend_keys = []
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        df_ds = df_agg[df_agg['dataset'] == ds].copy()
        
        plot_baselines(ax, df_ds)
        
        if mode == 'best':
            k_s = get_best(df_ds[df_ds['model_type'] == 'kernel'])
            gp_s = get_best(df_ds[df_ds['model_type'] == 'gp'])
            qf_k = get_best(df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['kernel_func'] != 'gp')])
            qf_gp = get_best(df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['kernel_func'] == 'gp')])
            cv_k = get_best(df_ds[(df_ds['model_type'] == 'conv') & (df_ds['kernel_func'] != 'gp')])
            cv_gp = get_best(df_ds[(df_ds['model_type'] == 'conv') & (df_ds['kernel_func'] == 'gp')])
            
            plot_series(ax, k_s, 'kernel', emphasize=True)
            plot_series(ax, gp_s, 'gp', emphasize=True)
            plot_series(ax, qf_k, 'q-former', emphasize=True)
            plot_series(ax, qf_gp, 'q-former-gp', emphasize=True)
            plot_series(ax, cv_k, 'conv', emphasize=True)
            plot_series(ax, cv_gp, 'conv-gp', emphasize=True)
            
            legend_keys = ['kernel', 'gp', 'q-former', 'q-former-gp', 'conv', 'conv-gp']

        elif mode == 'single':
            plot_series(ax, df_ds[df_ds['model_type'] == 'kernel'], 'kernel')
            plot_series(ax, df_ds[df_ds['model_type'] == 'gp'], 'gp')
            legend_keys = ['kernel', 'gp']

        elif mode == 'same':
            plot_series(ax, df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['bw_mode'] == 'Same') & (df_ds['kernel_func'] != 'gp')], 'q-former')
            plot_series(ax, df_ds[(df_ds['model_type'] == 'conv') & (df_ds['bw_mode'] == 'Same') & (df_ds['kernel_func'] != 'gp')], 'conv')
            plot_series(ax, df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['bw_mode'] == 'Same') & (df_ds['kernel_func'] == 'gp')], 'q-former-gp')
            plot_series(ax, df_ds[(df_ds['model_type'] == 'conv') & (df_ds['bw_mode'] == 'Same') & (df_ds['kernel_func'] == 'gp')], 'conv-gp')
            legend_keys = ['q-former', 'conv', 'q-former-gp', 'conv-gp']

        elif mode == 'diff':
            plot_series(ax, df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['bw_mode'] == 'Diff') & (df_ds['kernel_func'] != 'gp')], 'q-former')
            plot_series(ax, df_ds[(df_ds['model_type'] == 'conv') & (df_ds['bw_mode'] == 'Diff') & (df_ds['kernel_func'] != 'gp')], 'conv')
            plot_series(ax, df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['bw_mode'] == 'Diff') & (df_ds['kernel_func'] == 'gp')], 'q-former-gp')
            plot_series(ax, df_ds[(df_ds['model_type'] == 'conv') & (df_ds['bw_mode'] == 'Diff') & (df_ds['kernel_func'] == 'gp')], 'conv-gp')
            legend_keys = ['q-former', 'conv', 'q-former-gp', 'conv-gp']

        configure_axis(ax, title=ds)

    fig.supxlabel("Training Time (seconds)", fontsize=16, fontweight='bold', y=0.13)
    fig.supylabel("Error Rate", fontsize=16, fontweight='bold', x=0.06, y=0.55)
    
    create_global_legend(fig, legend_keys)
    
    plt.subplots_adjust(bottom=0.23, left=0.10, wspace=0.20)
    
    print(f"Saving {out_path}...")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    png_path = out_path.with_suffix('.png') 
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="experiment_results_add_original_forpaper/summary_results.csv")
    parser.add_argument("--out_dir", type=str, default="pictures")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return
        
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_raw = load_data(input_path)
    if df_raw.empty:
        print("Dataframe is empty.")
        return

    df_agg = get_aggregated(df_raw)
    
    plot_combined_figure(df_agg, 'best',   out_dir / "Figure_1_Best.pdf")
    plot_combined_figure(df_agg, 'single', out_dir / "Figure_2_Single.pdf")
    plot_combined_figure(df_agg, 'same',   out_dir / "Figure_3_Same.pdf")
    plot_combined_figure(df_agg, 'diff',   out_dir / "Figure_4_Diff.pdf")
    
    print("Done.")

if __name__ == "__main__":
    main()