import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict
from pathlib import Path
import ast

DATASETS = ["CharacterTrajectories", "SpokenArabicDigits", "UWaveGestureLibrary"]

MODEL_DISPLAY_NAMES = {
    'baseline_cubic': "Cubic",
    'baseline_linear': "Linear",
    'odernn': "ODE-RNN",
    'grud': "GRU-D",
    'log_ncde': "Log-NCDE",
    'kernel_gaussian': "Gaussian",
    'gp_gp': "GP",
    'qformer_gaussian': "MV (Gauss)",
    'qformer_gp': "MV (GP)",
    'conv_gaussian': "MVC (Gauss)",
    'conv_gp': "MVC (GP)"
}

PALETTE = sns.color_palette("tab10", n_colors=12)
MARKERS = ['o', 's', '^', 'v', 'D', 'X', 'P', '*', 'h', '>', '<', 'p']
STYLE_MAP = {name: {'color': PALETTE[i % len(PALETTE)], 'marker': MARKERS[i % len(MARKERS)]} 
             for i, name in enumerate(MODEL_DISPLAY_NAMES.values())}

def check_vals_match(saved_vals, target_vals):
    if not isinstance(saved_vals, (list, tuple, np.ndarray)): return False
    if not isinstance(target_vals, (list, tuple, np.ndarray)): return False
    
    s = np.array(saved_vals, dtype=float)
    t = np.array(target_vals, dtype=float)
    
    if s.size == 0 and t.size == 0: return True
    if s.size == 0 or t.size == 0: return False
    
    if s.shape == t.shape and np.allclose(s, t, atol=1e-4):
        return True
        
    if s.shape != t.shape: return False
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = s / t
        
    valid_ratios = ratios[np.isfinite(ratios)]
    if len(valid_ratios) != len(s): return False
        
    mean_ratio = np.mean(valid_ratios)
    if mean_ratio < 0.1: return False 
    
    is_scaled = np.allclose(valid_ratios, mean_ratio, rtol=0.2)
    return is_scaled

def extract_list_param(params, key_primary, key_secondary=None):
    val = params.get(key_primary)
    if val is None and key_secondary:
        val = params.get(key_secondary)
        
    if val is None: return []
    
    if isinstance(val, list): return val
    if isinstance(val, (float, int)): return [val]
    
    try:
        val_parsed = ast.literal_eval(str(val))
        if isinstance(val_parsed, list): return val_parsed
        return [val_parsed]
    except:
        return []

def is_target_config(dataset, model_type, params):
    tol = float(params.get('tol', 0))
    step_size = int(params.get('step_size', 0))
    
    bws_list = extract_list_param(params, 'bandwidths', 'bandwidth')
    ls_list = extract_list_param(params, 'length_scale', 'length_scales')

    if dataset == "CharacterTrajectories":
        if not np.isclose(tol, 0.001): return False, f"Tol mismatch: {tol} != 0.001"

        if model_type == 'log_ncde':
            if step_size != 10: return False, f"Step mismatch: {step_size} != 10"
        
        elif model_type == 'kernel_gaussian':
            if not check_vals_match(bws_list, [0.05]): return False, f"BW mismatch: {bws_list}"
        
        elif model_type == 'gp_gp':
            if not check_vals_match(ls_list, [0.6]): return False, f"LS mismatch: {ls_list} != [0.6]"
        
        elif 'qformer' in model_type or 'conv' in model_type:
            valid_targets = [[1.4, 1.4, 1.4, 1.4], [0.03, 0.1, 0.4, 1.4]]
            
            match_bw = any(check_vals_match(bws_list, t) for t in valid_targets)
            match_ls = any(check_vals_match(ls_list, t) for t in valid_targets)
            
            if not (match_bw or match_ls): 
                return False, f"Multi-Param mismatch. BW={bws_list}, LS={ls_list}"
            
        return True, "OK"

    elif dataset == "SpokenArabicDigits":
        if not np.isclose(tol, 0.001): return False, f"Tol mismatch: {tol} != 0.001"

        if model_type == 'log_ncde':
            if step_size != 20: return False, f"Step mismatch: {step_size} != 20"
        
        elif model_type == 'kernel_gaussian':
            if not check_vals_match(bws_list, [0.05]): return False, f"BW mismatch: {bws_list}"
        
        elif model_type == 'gp_gp':
            if not check_vals_match(ls_list, [0.6]): return False, f"LS mismatch: {ls_list} != [0.6]"
            
        elif 'qformer' in model_type or 'conv' in model_type:
            valid_targets = [[1.4, 1.4, 1.4], [0.03, 0.1, 0.4, 1.4], [0.1, 0.4, 1.4]]
            
            match_bw = any(check_vals_match(bws_list, t) for t in valid_targets)
            match_ls = any(check_vals_match(ls_list, t) for t in valid_targets)
            
            if not (match_bw or match_ls): 
                 return False, f"Multi-Param mismatch. BW={bws_list}, LS={ls_list}"

        return True, "OK"

    # --- UWAVE GESTURE LIBRARY ---
    elif dataset == "UWaveGestureLibrary":
        # Group 1: ODE-RNN
        if model_type == 'odernn':
            if np.isclose(tol, 0.01): return True, "OK"
            return False, f"Tol mismatch for ODE-RNN: {tol} != 0.01"
            
        # Group 2: GRU-D / Log-NCDE
        if model_type == 'grud':
            if np.isclose(tol, 0.001): return True, "OK"
            return False, f"Tol mismatch for GRU-D: {tol} != 0.001"
        if model_type == 'log_ncde':
            if np.isclose(tol, 0.001) and step_size == 30: return True, "OK"
            return False, f"Mismatch Log-NCDE: tol={tol}, step={step_size}"
            
        # Group 3: Baselines / CDEs
        if not np.isclose(tol, 0.0001): return False, f"Tol mismatch (expected 0.0001): {tol}"

        if model_type == 'kernel_gaussian':
            if not check_vals_match(bws_list, [0.05]): return False, f"BW mismatch: {bws_list}"
        
        elif model_type == 'gp_gp':
            if not check_vals_match(ls_list, [1.4]): return False, f"LS mismatch: {ls_list} != [1.4]"
            
        elif 'qformer' in model_type or 'conv' in model_type:
            valid_targets = [[0.03, 0.1, 0.4, 1.4], [0.05, 0.2, 0.6], [0.05, 0.05, 0.05, 0.05], [0.1, 0.2, 0.4]]
            
            match_bw = any(check_vals_match(bws_list, t) for t in valid_targets)
            match_ls = any(check_vals_match(ls_list, t) for t in valid_targets)
            
            if not (match_bw or match_ls): 
                return False, f"Multi-Param mismatch. BW={bws_list}, LS={ls_list}"
            
        return True, "OK"

    return False, "Unknown Dataset"

def get_model_key(data, filepath):    
    t = data.get('type', data.get('model_type', '')).lower()
    
    if not t:
        parts = filepath.parts
        found_tags = set()
        for part in parts:
            p = part.lower()
            if p == 'baseline': found_tags.add('baseline')
            elif p == 'odernn' or p == 'ode-rnn': found_tags.add('odernn')
            elif p == 'grud': found_tags.add('grud')
            elif p == 'log-ncde' or p == 'log_ncde': found_tags.add('log_ncde')
            elif p == 'kernel': found_tags.add('kernel')
            elif p == 'gp': found_tags.add('gp')
            elif p == 'qformer' or p == 'q-former': found_tags.add('qformer')
            elif p == 'conv': found_tags.add('conv')

        if 'qformer' in found_tags: t = 'qformer'
        elif 'conv' in found_tags: t = 'conv'
        elif 'odernn' in found_tags: t = 'odernn'
        elif 'grud' in found_tags: t = 'grud'
        elif 'log_ncde' in found_tags: t = 'log_ncde'
        elif 'kernel' in found_tags: t = 'kernel'
        elif 'gp' in found_tags: t = 'gp'
        elif 'baseline' in found_tags: t = 'baseline'

    if 'baseline' in t:
        interp = data.get('interpolation')
        if not interp: 
            if 'cubic' in str(filepath).lower(): interp = 'cubic'
            elif 'linear' in str(filepath).lower(): interp = 'linear'
            else: interp = 'linear'
        return f"baseline_{interp}"
    
    if t == 'odernn': return 'odernn'
    if t == 'grud': return 'grud'
    if t == 'log_ncde' or t == 'logncde': return 'log_ncde'
    
    # Kernel determination
    kernel = data.get('kernel')
    
    if not kernel:
        path_str = str(filepath).lower()
        if 'gaussian' in path_str: kernel = 'gaussian'
        elif 'gp' in path_str and t != 'gp': kernel = 'gp'
        else: kernel = 'gaussian'
    
    if t == 'gp': return "gp_gp"
    
    kernel = str(kernel).lower()
    
    if t == 'kernel': return f"kernel_{kernel}"
    
    if 'qformer' in t or 'q-former' in t: return f"qformer_{kernel}"
    if 'conv' in t: return f"conv_{kernel}"
        
    return "unknown"

def load_filtered_data(root_dir):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'nfe': [], 'acc': []})))
    
    files = glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True)
    print(f"Found {len(files)} JSON files. Scanning...")
    
    count_loaded = 0
    rejection_reasons = defaultdict(list)

    for fpath in files:
        if "metrics.json" in fpath: continue

        path_obj = Path(fpath)
        try:
            with open(fpath, 'r') as f: content = json.load(f)
        except Exception as e:
            continue
        
        # 1. Dataset Check
        dataset = content.get('dataset', content.get('dataset_name'))
        if not dataset:
            for d in DATASETS:
                if d in path_obj.parts:
                    dataset = d
                    break
        
        if not dataset or dataset not in DATASETS:
            continue

        # 2. Model Type Identification
        model_key = get_model_key(content, path_obj)
        if model_key == "unknown":
            rejection_reasons["Unknown Model Type"].append(fpath)
            continue
            
        if model_key not in MODEL_DISPLAY_NAMES:
            continue
        
        # 3. Config Verification
        is_valid, reason = is_target_config(dataset, model_key, content)
        if not is_valid:
            rejection_reasons[f"Config Reject ({model_key}) - {reason}"].append(fpath)
            continue
        
        # 4. Data Extraction
        display_name = MODEL_DISPLAY_NAMES[model_key]
        noise_res = content.get('noise_results', {})
        
        if not noise_res:
            continue
        
        count_loaded += 1
        for noise_val_str, metrics in noise_res.items():
            try:
                noise_val = float(noise_val_str)
                nfe = metrics.get('avg_nfe', 0.0)
                acc = metrics.get('accuracy', 0.0)
                
                data[dataset][display_name][noise_val]['nfe'].append(nfe)
                data[dataset][display_name][noise_val]['acc'].append(acc)
            except: pass
            
    print("="*60)
    print(f"SUCCESS: Loaded {count_loaded} valid files.")
    print("="*60)
    
    print("DEBUG: Rejection Statistics (Top 3 examples per reason):")
    for reason, examples in rejection_reasons.items():
        print(f"\n[x] {reason} (Total: {len(examples)})")
        for ex in examples[:3]:
            print(f"    - {ex}")
    print("="*60)
    
    return data

def plot_noise_robustness(root_dir, output_dir):
    data = load_filtered_data(root_dir)
    if not data:
        print(f"No valid data found in {root_dir}.")
        return

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex=True)
    fig.suptitle("Noise Robustness Analysis", fontsize=24, fontweight='bold', y=0.97)
    
    all_found_models = set()
    for d in data: all_found_models.update(data[d].keys())
    
    predefined_order = list(MODEL_DISPLAY_NAMES.values())
    sorted_models = [m for m in predefined_order if m in all_found_models]
    sorted_models += [m for m in all_found_models if m not in sorted_models]

    for col, dataset in enumerate(DATASETS):
        ax_nfe = axes[0, col]
        ax_err = axes[1, col]
        
        if dataset not in data:
            for ax in [ax_nfe, ax_err]: ax.set_visible(False)
            continue
            
        ds_data = data[dataset]
        ax_nfe.set_title(dataset, fontweight='bold', pad=15)
        
        for model_name in sorted_models:
            if model_name not in ds_data: continue
                
            noise_levels = sorted(ds_data[model_name].keys())
            if not noise_levels: continue
            
            if 0.0 in ds_data[model_name]:
                base_nfes = ds_data[model_name][0.0]['nfe']
                base_nfe = np.mean(base_nfes) if base_nfes else 1.0
            else:
                base_nfes = ds_data[model_name][noise_levels[0]]['nfe']
                base_nfe = np.mean(base_nfes) if base_nfes else 1.0
            if base_nfe < 1e-6: base_nfe = 1.0

            x_vals = []
            y_nfe = []
            y_err = []
            
            for nl in noise_levels:
                nfes = ds_data[model_name][nl]['nfe']
                accs = ds_data[model_name][nl]['acc']
                
                if not nfes or not accs: continue
                
                x_vals.append(nl)
                y_nfe.append(np.mean(nfes) / base_nfe)
                y_err.append(1.0 - (np.mean(accs) / 100.0))
            
            style = STYLE_MAP.get(model_name, {'color': 'black', 'marker': 'o'})
            
            ax_nfe.plot(x_vals, y_nfe, label=model_name, **style, alpha=0.8)
            ax_err.plot(x_vals, y_err, label=model_name, **style, alpha=0.8)

        ax_nfe.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_err.grid(True, which='both', linestyle='--', alpha=0.5)
        
        if col == 0:
            ax_nfe.set_ylabel("Relative NFE", fontweight='bold')
            ax_err.set_ylabel("Error Rate", fontweight='bold')
            
        ax_err.set_xlabel("Noise Level (std)", fontweight='bold')

    handles, labels = [], []
    for model in sorted_models:
        style = STYLE_MAP.get(model, {'color': 'black', 'marker': 'o'})
        h = plt.Line2D([0], [0], **style, linestyle='-')
        handles.append(h)
        labels.append(model)
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=6, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15, wspace=0.15, hspace=0.1)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "noise_robustness.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_root", type=str, default="experiment_best", 
                        help="Root dir containing dataset folders with noise results")
    parser.add_argument("--output", type=str, default="pictures",
                        help="Output directory for plots")
    args = parser.parse_args()

    plot_noise_robustness(args.plots_root, args.output)

if __name__ == "__main__":
    main()