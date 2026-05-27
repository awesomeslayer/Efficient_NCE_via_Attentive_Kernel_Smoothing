import pandas as pd
import numpy as np
import logging
import sys
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def format_mean_std(mean, std):
    if pd.isna(mean): return "-"
    if pd.isna(std): std = 0.0
    return f"{mean:.2f} ± {std:.2f}"

def format_float(val):
    if pd.isna(val): return "-"
    return f"{val:.2f}"

def get_param_signature(row):
    params = []
    m_type = row.get('model_type', '')

    bw_str = str(row.get('bandwidths_str', 'N/A'))
    if bw_str not in ['N/A', 'nan', 'None']:
        params.append(f"BW:{bw_str}")

    noise = str(row.get('noise_std', 'N/A'))
    if noise not in ['N/A', 'nan', 'None']:
        params.append(f"Noise:{noise}")
        
    ls = str(row.get('length_scale', 'N/A'))
    if ls not in ['N/A', 'nan', 'None']:
        params.append(f"LS:{ls}")

    ks = str(row.get('conv_kernel_size', 'N/A'))
    if ks not in ['N/A', 'nan', 'None']: 
        params.append(f"KSize:{ks}")
        
    aggr = str(row.get('aggregation', 'N/A'))
    if aggr not in ['N/A', 'nan', 'None', 'concat']: 
        params.append(f"Agg:{aggr}")

    step = str(row.get('step_size', 'N/A'))
    if step not in ['N/A', 'nan', 'None']:
        params.append(f"Step:{step}")
        
    depth = str(row.get('depth', 'N/A'))
    if depth not in ['N/A', 'nan', 'None']:
        params.append(f"Depth:{depth}")

    ts = str(row.get('time_scaling', 'no'))
    if ts != 'no':
        params.append(f"TS:{ts}")
        
    add_time = str(row.get('add_time', 'yes')).lower()
    if add_time in ['false', 'no', '0']:
        params.append("NoTimeCh") 

    try:
        tol = float(row.get('tolerance', 0.0))
        if tol > 0 and tol != 1e-3 and tol != 0.001: 
            params.append(f"Tol:{tol}")
    except:
        pass

    return " | ".join(params) if params else "Standard"

def generate_table(input_csv):
    input_path = Path(input_csv)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    group_cols = [
        'dataset', 'model_type', 'interpolation', 'kernel_func', 
        'bandwidths_str', 'bw_mode', 'time_scaling', 'add_time',
        'tolerance', 'noise_std', 'length_scale', 'conv_kernel_size', 
        'aggregation', 'step_size', 'depth'
    ]
    
    valid_group_cols = [c for c in group_cols if c in df.columns]
    
    for col in valid_group_cols:
        df[col] = df[col].fillna('N/A').astype(str)

    agg_dict = {
        'test_acc': ['mean', 'std'],
        'avg_nfe_test': ['mean', 'std'],
        'train_all_time': ['mean'],
        'test_all_time': ['mean'],
        'seed': 'count'
    }
    
    logger.info("Grouping and aggregating results...")
    grouped = df.groupby(valid_group_cols, as_index=False).agg(agg_dict)
    
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
    
    summary_rows = []
    
    for _, row in grouped.iterrows():
        m_type = row.get('model_type', 'Unknown')
        interp = row.get('interpolation', 'N/A')
        kern = row.get('kernel_func', 'N/A')
        
        display_name = m_type.upper()
        
        if m_type == 'baseline':
            display_name = f"Baseline ({interp})"
        elif m_type == 'odernn':
            display_name = "ODE-RNN"
        elif m_type == 'grud':
            display_name = "GRU-D"
        elif m_type == 'log_ncde':
            display_name = "Log-NCDE"
        elif m_type in ['qformer', 'conv']:
            display_name = f"{m_type.capitalize()} ({kern})"
        elif m_type == 'kernel':
            display_name = f"Kernel ({kern})"
        elif m_type == 'gp':
            display_name = "GP-CDE"

        param_str = get_param_signature(row)

        res = {
            'Dataset': row.get('dataset', '-'),
            'Model': display_name,
            'Params': param_str,
            'Seeds': row['seed_count'],
            'Acc (%)': format_mean_std(row['test_acc_mean'], row['test_acc_std']),
            'NFE': format_mean_std(row['avg_nfe_test_mean'], row['avg_nfe_test_std']),
            'Train(s)': format_float(row['train_all_time_mean']),
            'Test(s)': format_float(row['test_all_time_mean']),
            
            
            '_sort_acc': row['test_acc_mean']
        }
        summary_rows.append(res)

    final_df = pd.DataFrame(summary_rows)
    
    if final_df.empty:
        logger.warning("No results to save.")
        return

    final_df = final_df.sort_values(
        by=['Dataset', 'Model', '_sort_acc'], 
        ascending=[True, True, False]
    )
    final_df = final_df.drop(columns=['_sort_acc'])
    
    output_file = input_path.parent / "final_summary_table.csv"
    final_df.to_csv(output_file, index=False)
    
    logger.info("=" * 40)
    logger.info(f"Summary Table Generated: {output_file}")
    logger.info("=" * 40)
    print(final_df.head(10).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create final summary table from aggregated CSV.")
    parser.add_argument("csv_file", help="Path to the summary_results.csv file")
    args = parser.parse_args()

    generate_table(args.csv_file)