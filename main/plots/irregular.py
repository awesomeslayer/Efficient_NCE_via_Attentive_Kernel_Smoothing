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

STYLE = {
    'baseline-linear': {'color': 'tab:cyan',   'marker': 'd', 'label': 'Linear'},
    'baseline-cubic':  {'color': 'tab:olive',  'marker': 'X', 'label': 'Cubic'},
    'odernn':          {'color': 'tab:pink',   'marker': 'h', 'label': 'ODE-RNN'}, 
    'grud':            {'color': 'black',      'marker': '^', 'label': 'GRU-D'},   
    'log_ncde':        {'color': 'gray',       'marker': 'v', 'label': 'Log-NCDE'},
    'kernel':          {'color': 'tab:blue',   'marker': 'o', 'label': 'Gaussian CDE'},
    'gp':              {'color': 'tab:orange', 'marker': 'p', 'label': 'GP CDE'},
    'q-former':        {'color': 'tab:green',  'marker': 's', 'label': 'MV-CDE (Gaussian)'},
    'conv':            {'color': 'tab:purple', 'marker': '*', 'label': 'MVC-CDE (Gaussian)'},
    'q-former-gp':     {'color': 'tab:red',    'marker': '<', 'label': 'MV-CDE (GP)'},
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
        df['model_type'] = df['model_type'].replace({
            'qformer': 'q-former', 
            'logncde': 'log_ncde'
        })

    for c in ['test_acc', 'train_time_sec', 'seed', 'drop_rate']:
        if c in df.columns: 
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    if 'drop_rate' in df.columns:
        df['drop_rate'] = df['drop_rate'].fillna(0.0)
    else:
        df['drop_rate'] = 0.0
        
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
                  'time_scaling', 'aggregation', 'drop_rate']
    
    valid = [c for c in group_cols if c in df.columns]
    agg = df.groupby(valid, dropna=False)[['test_error', 'train_time_sec']].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(c).strip() if c[1] else c[0] for c in agg.columns.values]
    return agg

def plot_series(ax, df_sub, style_key):
    if df_sub.empty: return
    s = STYLE.get(style_key, {'color': 'k', 'marker': '.', 'label': style_key})
    
    x = df_sub['train_time_sec_mean']
    y = df_sub['test_error_mean']
    x_err = df_sub['train_time_sec_std'].fillna(0)
    y_err = df_sub['test_error_std'].fillna(0)
    
    alpha_points = 0.8 
    alpha_bars = 0.3
    size = 80
    zorder = 5
    edge_c = 'white'
    lw = 0.8

    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', 
                ecolor=s['color'], alpha=alpha_bars, elinewidth=1.5, capsize=0, zorder=zorder-1)
    
    ax.scatter(x, y, c=s['color'], marker=s['marker'], s=size, 
               edgecolors=edge_c, linewidth=lw, alpha=alpha_points, zorder=zorder)

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
    
    seen = set()
    unique_keys = [x for x in mode_keys if not (x in seen or seen.add(x))]

    for k in unique_keys:
        s = STYLE[k]
        h = mlines.Line2D([], [], color=s['color'], marker=s['marker'], 
                          linestyle='None', markersize=10, label=s['label'])
        handles.append(h)
        labels.append(s['label'])
        
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
               ncol=4, frameon=False, columnspacing=1.5)

def plot_droprate_figure(df_agg, dr, out_path):
    datasets = sorted(df_agg['dataset'].unique())
    n_ds = len(datasets)
    if n_ds == 0: return

    df_dr = df_agg[df_agg['drop_rate'] == dr]
    if df_dr.empty: return

    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 6), sharex=False, sharey=False)
    if n_ds == 1: axes = [axes]
    
    plotted_keys = set()
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        df_ds = df_dr[df_dr['dataset'] == ds].copy()
        
        def _plot(subset, style_key):
            if not subset.empty:
                plot_series(ax, subset, style_key)
                plotted_keys.add(style_key)

        # Baseline
        _plot(df_ds[(df_ds['model_type'] == 'baseline') & (df_ds['interpolation'] == 'linear')], 'baseline-linear')
        _plot(df_ds[(df_ds['model_type'] == 'baseline') & (df_ds['interpolation'] == 'cubic')], 'baseline-cubic')
        _plot(df_ds[df_ds['model_type'] == 'odernn'], 'odernn')
        _plot(df_ds[df_ds['model_type'] == 'grud'], 'grud')
        _plot(df_ds[df_ds['model_type'] == 'log_ncde'], 'log_ncde')
        
        # Proposed models
        _plot(df_ds[df_ds['model_type'] == 'kernel'], 'kernel')
        _plot(df_ds[df_ds['model_type'] == 'gp'], 'gp')
        _plot(df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['kernel_func'] != 'gp')], 'q-former')
        _plot(df_ds[(df_ds['model_type'] == 'q-former') & (df_ds['kernel_func'] == 'gp')], 'q-former-gp')
        _plot(df_ds[(df_ds['model_type'] == 'conv') & (df_ds['kernel_func'] != 'gp')], 'conv')
        _plot(df_ds[(df_ds['model_type'] == 'conv') & (df_ds['kernel_func'] == 'gp')], 'conv-gp')

        configure_axis(ax, title=f"{ds} (Drop Rate: {dr*100:.0f}%)")

    fig.supxlabel("Training Time (seconds)", fontsize=16, fontweight='bold', y=0.18)
    fig.supylabel("Error Rate", fontsize=16, fontweight='bold', x=0.06, y=0.60)
    
    ordered_keys = ['baseline-linear', 'baseline-cubic', 'odernn', 'grud', 'log_ncde', 
                    'kernel', 'gp', 'q-former', 'conv', 'q-former-gp', 'conv-gp']
    legend_keys = [k for k in ordered_keys if k in plotted_keys]
    
    create_global_legend(fig, legend_keys)
    
    plt.subplots_adjust(bottom=0.30, left=0.10, wspace=0.20)
    
    print(f"Saving {out_path}...")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    png_path = out_path.with_suffix('.png') 
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to summary_results.csv")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return
        
    out_dir = input_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_raw = load_data(input_path)
    if df_raw.empty:
        print("Dataframe is empty.")
        return

    df_agg = get_aggregated(df_raw)
    
    drop_rates = sorted(df_agg['drop_rate'].unique())
    print(f"Found drop rates: {drop_rates}")
    
    for dr in drop_rates:
        dr_str = str(dr).replace('.', '_')
        plot_droprate_figure(df_agg, dr, out_dir / f"Scatter_Time_vs_Error_dr-{dr_str}.pdf")
    
    print("Done.")

if __name__ == "__main__":
    main()