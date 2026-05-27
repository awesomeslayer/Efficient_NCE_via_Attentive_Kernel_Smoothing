import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_bandwidths(d):
    if 'bandwidths' in d and d['bandwidths'] is not None:
        return str(d['bandwidths'])
    if 'bandwidth' in d and d['bandwidth'] is not None:
        return str(d['bandwidth'])
    if 'bw_multiplier' in d and d['bw_multiplier'] is not None:
        return f"mult:{d['bw_multiplier']}"
    return "N/A"

def infer_model_type(filepath):
    parts = filepath.parts
    known = {'ODE-RNN': 'odernn', 'GRU-D': 'grud', 'Log-NCDE': 'log_ncde', 
             'baseline': 'baseline', 'kernel': 'kernel', 'GP': 'gp', 
             'qformer': 'qformer', 'conv': 'conv', 'Mamba': 'mamba', 'mamba': 'mamba'}
    for part in parts:
        if part in known: return known[part]
    return "unknown"

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None

    def get_val(key, default='N/A'):
        val = data.get(key, default)
        return val if val is not None else 'N/A'

    row = {
        'dataset': get_val('dataset', filepath.parts[-4] if len(filepath.parts)>=4 else 'Unknown'),
        'seed': get_val('seed', 0),
        'model_type': infer_model_type(filepath),
        
        'drop_rate': float(get_val('drop_rate', 0.0)),
        'time_scaling': str(get_val('ts_factor', 'no')),
        'add_time': str(get_val('add_time', True)).lower(),
        'tolerance': get_val('tol', 'N/A'),
        
        'interpolation': get_val('interpolation'),
        'kernel_func': get_val('kernel'),
        'conv_kernel_size': get_val('conv_kernel_size'),
        'depth': get_val('depth'),
        'step_size': get_val('step_size'),
        'length_scale': get_val('length_scale'),
        'noise_std': get_val('noise_std'),
        'bandwidths_str': parse_bandwidths(data),
        'smoothing_factor': get_val('smoothing_factor', 'N/A'),
        
        'regularization': get_val('regularization', 'N/A'),
        'n_layers': get_val('n_layers', 'N/A'),
        'd_state': get_val('d_state', 'N/A'),
        'd_conv': get_val('d_conv', 'N/A'),
        'expand_factor': get_val('expand_factor', 'N/A'),
        'pscan': get_val('pscan', 'N/A'),
        
        'num_params': get_val('num_params', np.nan),
        'peak_vram_mb': get_val('peak_vram_mb', np.nan),
        'test_acc': get_val('final_test_accuracy', np.nan),
        'avg_nfe_test': get_val('avg_test_test', np.nan),
        'train_all_time': get_val('train_all_time', np.nan),
        'test_all_time': get_val('test_all_time', np.nan)
    }
    return row

def format_mean_std(mean, std):
    if pd.isna(mean): return "-"
    if pd.isna(std) or std == 0.0: return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"

def format_float(val):
    if pd.isna(val): return "-"
    return f"{val:.2f}"

def format_int(val):
    if pd.isna(val): return "-"
    return f"{int(val):,}"

def get_model_name(row):
    m = str(row.get('model_type', '')).lower()
    interp = str(row.get('interpolation', 'N/A'))
    kern = str(row.get('kernel_func', 'N/A')).capitalize()

    if m == 'baseline': return f"Baseline ({interp})"
    if m == 'odernn': return "ODE-RNN"
    if m == 'grud': return "GRU-D"
    if m in ['log_ncde', 'logncde']: return "Log-NCDE"
    if m == 'kernel': return f"Kernel ({kern})"
    if m == 'gp': return "GP-CDE"
    if m == 'qformer': return f"Q-Former ({kern})"
    if m == 'conv': return f"ConvCDE ({kern})"
    if m == 'mamba': return "Mamba" 
    
    return m.upper()

def get_param_signature(row):
    params = []
    
    ts = str(row.get('time_scaling', 'no'))
    if ts != 'no' and ts != 'N/A': params.append(f"TS:{ts}")
        
    tol = str(row.get('tolerance', 'N/A'))
    if tol != 'N/A': params.append(f"Tol:{tol}")

    bw = str(row.get('bandwidths_str', 'N/A'))
    if bw not in ['N/A', 'nan', 'None']: params.append(f"BW:{bw}")
        
    ls = str(row.get('length_scale', 'N/A'))
    if ls not in ['N/A', 'nan', 'None']: params.append(f"LS:{ls}")

    noise = str(row.get('noise_std', 'N/A'))
    if noise not in ['N/A', 'nan', 'None']: params.append(f"Noise:{noise}")
        
    ks = str(row.get('conv_kernel_size', 'N/A'))
    if ks not in ['N/A', 'nan', 'None']: params.append(f"KSize:{ks}")

    step = str(row.get('step_size', 'N/A'))
    if step not in ['N/A', 'nan', 'None']: params.append(f"Step:{step}")
        
    depth = str(row.get('depth', 'N/A'))
    if depth not in ['N/A', 'nan', 'None']: params.append(f"Depth:{depth}")

    at = str(row.get('add_time', 'true'))
    if at in ['false', 'no']: params.append("NoTime")

    sf = str(row.get('smoothing_factor', 'N/A'))
    if sf not in ['N/A', 'nan', 'None']: params.append(f"S:{sf}")
        
    n_layers = str(row.get('n_layers', 'N/A'))
    if n_layers not in ['N/A', 'nan', 'None']: params.append(f"L:{n_layers}")
        
    d_state = str(row.get('d_state', 'N/A'))
    if d_state not in ['N/A', 'nan', 'None']: params.append(f"DSt:{d_state}")
        
    expand = str(row.get('expand_factor', 'N/A'))
    if expand not in ['N/A', 'nan', 'None']: params.append(f"Exp:{expand}")
        
    pscan = str(row.get('pscan', 'N/A'))
    if pscan not in ['N/A', 'nan', 'None']: params.append(f"Pscan:{pscan}")
        
    return " | ".join(params) if params else "Standard"


def aggregate_directory(target_dir):
    base_path = Path(target_dir)
    if not base_path.exists():
        logger.error(f"Directory not found: {base_path}")
        return

    logger.info(f"Scanning directory: {base_path}")
    json_files = list(base_path.rglob("*.json"))

    if not json_files:
        logger.warning("No JSON files found!")
        return

    rows = [process_file(jf) for jf in json_files]
    df = pd.DataFrame([r for r in rows if r])

    numeric_cols = [
        'test_acc', 'avg_nfe_test', 'train_all_time', 
        'test_all_time', 'drop_rate', 'num_params', 'peak_vram_mb'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    raw_path = base_path / "summary_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw dump to {raw_path.name}")

    group_cols = [
        'dataset', 'drop_rate', 'model_type', 'interpolation','smoothing_factor', 
        'kernel_func', 'bandwidths_str', 'time_scaling', 'add_time', 'tolerance', 
        'noise_std', 'length_scale', 'conv_kernel_size', 'step_size', 'depth',
    
        'n_layers', 'd_state', 'd_conv', 'expand_factor', 'pscan'
    ]
    
    for col in group_cols:
        if col in df.columns:
            df[col] = df[col].fillna('N/A').astype(str)

    agg_dict = {
        'test_acc': ['mean', 'std'],
        'avg_nfe_test': ['mean', 'std'],
        'train_all_time': ['mean', 'std'], 
        'test_all_time': ['mean', 'std'],  
        'num_params': ['mean'],
        'peak_vram_mb': ['mean'],
        'seed': 'count'
    }

    grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]

    unique_drs = grouped['drop_rate'].unique()

    for dr in unique_drs:
        dr_df = grouped[grouped['drop_rate'] == dr].copy()
        
        summary_rows = []
        for _, row in dr_df.iterrows():
            summary_rows.append({
                'Dataset': row['dataset'],
                'Model': get_model_name(row),
                'Params Signature': get_param_signature(row),
                'Seeds': row['seed_count'],
                'Acc (%)': format_mean_std(row['test_acc_mean'], row['test_acc_std']),
                'NFE': format_mean_std(row['avg_nfe_test_mean'], row['avg_nfe_test_std']),
                'Model Params': format_int(row['num_params_mean']),
                'Peak VRAM (MB)': format_float(row['peak_vram_mb_mean']),
                
                'Train(s)': format_mean_std(row['train_all_time_mean'], row['train_all_time_std']),
                'Test(s)': format_mean_std(row['test_all_time_mean'], row['test_all_time_std']),
                
                '_sort_acc': row['test_acc_mean']
            })

        final_df = pd.DataFrame(summary_rows)
        if final_df.empty: continue

        final_df = final_df.sort_values(by=['Dataset', '_sort_acc'], ascending=[True, False])
        final_df = final_df.drop(columns=['_sort_acc'])

        dr_clean = str(dr).replace('.', '_')
        out_csv = base_path / f"final_table_dr-{dr_clean}.csv"
        
        final_df.to_csv(out_csv, index=False)
        logger.info("-" * 50)
        logger.info(f"Generated Table for Drop Rate = {dr} -> {out_csv.name}")
        
        print(final_df.head(10).to_string(index=False))
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Neural CDE results into clean tables.")
    parser.add_argument("dirs", nargs='+', help="List of experiment directories (e.g., experiment_irregular/)")
    args = parser.parse_args()

    print("="*60)
    print("      AGGREGATING EXPERIMENT RESULTS BY DROP RATE")
    print("="*60)

    for d in args.dirs:
        aggregate_directory(d)

    print("Done.")