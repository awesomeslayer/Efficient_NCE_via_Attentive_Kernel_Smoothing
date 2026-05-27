import yaml
import os
import json
import argparse
import numpy as np
from pathlib import Path
import torch
from main.loading.utils import set_seed, setup_logger
from main.loading.data_loader import get_data, apply_drop_rate
from main.model.model import NeuralCDE, QFormerCDE, ConvCDE, ODERNN, GRUD, LogNeuralCDE, MambaModel
from main.model.trainer import run_experiment

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def merge_params(base, specific):
    p = base.copy()
    p.update(specific)
    return p

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def main():
    parser = argparse.ArgumentParser(description="Clean NeuralCDE Experiments")
    parser.add_argument("--config", type=str, default="./main/config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    logger = setup_logger(cfg['log_dir'])
    
    logger.info("=== STARTED EXPERIMENTS ===")
    
    ts_factors = cfg.get('time_scaling_factors', [])
    if not ts_factors:
        ts_factors = [cfg.get('time_scaling_factor', 'no')]

    add_time_params = cfg.get('add_time', 'yes')
    if not isinstance(add_time_params, list):
        add_time_params = [add_time_params]
    
    # ---------------------------------------------------
    # LOOP 1: SEEDS
    # ---------------------------------------------------
    for seed in cfg['seeds']:
        set_seed(seed)
        cfg['seed'] = seed 

        # -----------------------------------------------------------
        # LOOP 2: TIME SCALING
        # -----------------------------------------------------------
        
        for ts_factor in ts_factors:
            cfg['time_scaling_factor'] = ts_factor
            ts_name = "ts-no" if str(ts_factor).lower() == 'no' else f"ts-{ts_factor}"
            
            # -------------------------------------------------------
            # LOOP 3: ADD TIME (YES/NO)
            # -------------------------------------------------------
            for add_time_val in add_time_params:
                cfg['add_time'] = add_time_val
                add_time_bool = (str(add_time_val).lower() == 'yes')
                
                time_str = "time-yes" if add_time_bool else "time-no"

                logger.info(f"\n>>> PROCESSING: {ts_name} | {time_str}")

                clean_data_tuple = get_data(cfg, logger)
                clean_train_X, clean_val_X, clean_test_X, train_y, val_y, test_y, t_grid = clean_data_tuple
                
                # Retrieve drop rates
                test_dr = cfg.get('test_drop_rates', False)
                drop_rates = cfg.get('drop_rates', [0.0]) if test_dr else [0.0]
                
                
                # LOOP 3.5: REGULARIZATION (Attention Diversity)
                # -------------------------------------------------------
                reg_params = cfg.get('regularization', [False])
                if not isinstance(reg_params, list):
                    reg_params = [reg_params]

                for use_reg in reg_params:
                    cfg['regularization'] = use_reg
                    reg_str = "reg-on" if use_reg else "reg-off"
                
                    # -----------------------------------------------------------
                    # LOOP 4: DROP RATES (MISSING DATA)
                    # -----------------------------------------------------------
                    for drop_rate in drop_rates:
                        dr_str = f"dr-{drop_rate}" if drop_rate > 0 else "dr-0.0"
                        logger.info(f"\n>>> PROCESSING: {ts_name} | {time_str} | {dr_str}")
                        
                        # Apply simulated missingness
                        train_X = apply_drop_rate(clean_train_X, drop_rate, add_time_val)
                        val_X = apply_drop_rate(clean_val_X, drop_rate, add_time_val)
                        test_X = apply_drop_rate(clean_test_X, drop_rate, add_time_val)
                        
                        data_tuple = (train_X, val_X, test_X, train_y, val_y, test_y, t_grid)
                        needs_mask = torch.isnan(train_X).any().item()
                        seq_len = train_X.shape[1]
                        input_channels = train_X.shape[2] 
                        output_channels = len(torch.unique(train_y))
                        bw_multiplier = float(seq_len)
                        model_t_grid = t_grid if not add_time_bool else None
                        dataset_res_dir = Path(cfg['results_dir']) / cfg['dataset_name'] / dr_str
                        
                        
                            
                        # ------------------------------------------------
                        # LOOP 5: TOLERANCE
                        # ------------------------------------------------
                        for tol in cfg['tolerances']:
                            
                            base_info = {
                                'dataset': cfg['dataset_name'],
                                'seed': seed,
                                'tol': tol,
                                'ts_factor': ts_factor,
                                'add_time': add_time_bool,  
                                'bw_multiplier': bw_multiplier,
                                'drop_rate': drop_rate,
                                'batch_size': cfg['batch_size'],
                                'num_epochs': cfg['num_epochs'],
                                'lr': cfg['lr'],
                                'regularization': use_reg, 
                            }
                        
                            logger.info(f"--- Seed: {seed} | Tol: {tol} | {ts_name} | {time_str} ---")

                            # ===============================================
                            # EXPERIMENT A: BASELINE (Linear / Cubic / Smoothing Spline)
                            # ===============================================
                            if cfg['experiments_to_run'].get('baseline', False):
                                for interp in cfg['baseline_params']['interpolations']:
                                    
                                    if interp == 'smoothing_spline':
                                        s_factors = cfg['baseline_params'].get('smoothing_factors', [0.1, 0.5, 1.0])
                                    else:
                                        s_factors = [None] 
                                        
                                    for s_factor in s_factors:
                                        if s_factor is not None:
                                            exp_name = f"baseline_{interp}_s-{s_factor}_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                            kp = {'tol': tol, 'smoothing_factor': s_factor}
                                        else:
                                            exp_name = f"baseline_{interp}_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                            kp = {'tol': tol}
                                            
                                        out_path = dataset_res_dir / "baseline" / f"{exp_name}_seed-{seed}.json"
                                        
                                        if not out_path.exists():
                                            model = NeuralCDE(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                            interpolation=interp, tol=tol, 
                                                            add_time=add_time_bool, t_grid=model_t_grid)
                                            
                                            res = run_experiment(model, data_tuple, cfg, logger, exp_name, interp, kp, dr=drop_rate)
                                            
                                            res.update(base_info)
                                            res['interpolation'] = interp
                                            if s_factor is not None:
                                                res['smoothing_factor'] = s_factor
                                                
                                            save_json(out_path, res)

                            # ===============================================
                            # EXPERIMENT B: KERNEL (Single)
                            # ===============================================
                            if cfg['experiments_to_run'].get('kernel', False):
                                k_cfg = cfg['kernel_params']
                                for k_name in k_cfg['kernels']:
                                    for bw_raw in k_cfg['bandwidths']:
                                        bw_val = round(bw_raw * bw_multiplier, 4)
                                        k_params = {'kernel': k_name, 'bandwidth': bw_val, 'tol': tol}
                                        
                                        exp_name = f"kernel-{k_name}_bw-{bw_val}_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                        out_path = dataset_res_dir / "kernel" / k_name / f"{exp_name}_seed-{seed}.json"
                                        
                                        if not out_path.exists():
                                            model = NeuralCDE(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                            "kernel", kernel_params=k_params, tol=tol, 
                                                            add_time=add_time_bool, t_grid=model_t_grid)
                                            
                                            res = run_experiment(model, data_tuple, cfg, logger, exp_name, "kernel", k_params, dr=drop_rate)
                                            res.update(base_info)
                                            res.update(k_params)
                                            save_json(out_path, res)

                            # ===============================================
                            # EXPERIMENT C: GP (Single)
                            # ===============================================
                            if cfg['experiments_to_run'].get('gp', False):
                                g_cfg = cfg['gp_params']
                                for ls_raw in g_cfg['length_scales']:
                                    ls_val = round(ls_raw * bw_multiplier, 4)
                                    for noise in g_cfg['noise_stds']:
                                        gp_p = {'length_scale': ls_val, 'noise_std': noise, 'tol': tol}
                                        
                                        exp_name = f"gp_ls-{ls_val}_noise-{noise}_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                        out_path = dataset_res_dir / "GP" / f"{exp_name}_seed-{seed}.json"
                                        
                                        if not out_path.exists():
                                            model = NeuralCDE(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                            "gp", kernel_params=gp_p, tol=tol, 
                                                            add_time=add_time_bool, t_grid=model_t_grid)
                                            
                                            res = run_experiment(model, data_tuple, cfg, logger, exp_name, "gp", gp_p, dr=drop_rate)
                                            res.update(base_info)
                                            res.update(gp_p)
                                            save_json(out_path, res)

                            # ===============================================
                            # EXPERIMENT D: ODE-RNN & GRU-D
                            # ===============================================
                            if cfg['experiments_to_run'].get('odernn', False):
                                # Added {time_str} to filename
                                exp_name = f"odernn_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                out_path = dataset_res_dir / "ODE-RNN" / f"{exp_name}_seed-{seed}.json"
                                
                                if not out_path.exists():
                                    model = ODERNN(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                tol=tol, add_time=add_time_bool, t_grid=model_t_grid,use_mask=needs_mask)
                                    res = run_experiment(model, data_tuple, cfg, logger, exp_name, "odernn", {'tol': tol}, dr=drop_rate)                                
                                    res.update(base_info)
                                    save_json(out_path, res)
                            
                            if cfg['experiments_to_run'].get('grud', False):
                                exp_name = f"grud_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                out_path = dataset_res_dir / "GRU-D" / f"{exp_name}_seed-{seed}.json"
                                
                                if not out_path.exists():
                                    model = GRUD(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                add_time=add_time_bool, t_grid=model_t_grid,use_mask=needs_mask)
                                    res = run_experiment(model, data_tuple, cfg, logger, exp_name, "grud", {'tol': tol}, dr=drop_rate)
                                    res.update(base_info)
                                    save_json(out_path, res)

                            # ===============================================
                            # EXPERIMENT: MAMBA (Solver-Free Baseline)
                            # ===============================================
                            if cfg['experiments_to_run'].get('mamba', False):
                                exp_name = f"mamba_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                out_path = dataset_res_dir / "Mamba" / f"{exp_name}_seed-{seed}.json"
                                
                                if not out_path.exists():
                                    os.makedirs(out_path.parent, exist_ok=True)
                                    
                                    m_params = cfg.get('mamba_params', {})
                                    
                                    model = MambaModel(
                                        input_channels=input_channels, 
                                        hidden_channels=cfg['hidden_dim'], 
                                        output_channels=output_channels, 
                                        seq_len=seq_len, 
                                        add_time=add_time_bool, 
                                        t_grid=model_t_grid,
                                        use_mask=needs_mask,
                                        mamba_params=m_params 
                                    )
                                    
                                    res = run_experiment(model, data_tuple, cfg, logger, exp_name, "mamba", {'tol': tol}, dr=drop_rate)
                                    res.update(base_info)
                                    res.update(m_params) 
                                    save_json(out_path, res)

                            # ===============================================
                            # EXPERIMENT: LOG-NCDE
                            # ===============================================
                            if cfg['experiments_to_run'].get('log_ncde', False):
                                lncde_cfg = cfg.get('log_ncde_params', {'step_sizes': [5], 'depths': [1]})
                                
                                steps = lncde_cfg.get('step_sizes', [5])
                                depths = lncde_cfg.get('depths', [1])

                                for step in steps:
                                    for depth in depths:
                                        if step > seq_len:
                                            logger.warning(f"Skipping LogNCDE step={step} (larger than seq_len={seq_len})")
                                            continue

                                        log_params = {'step_size': step, 'depth': depth, 'tol': tol}
                                        
                                        exp_name = f"logncde_depth{depth}_step{step}_tol-{tol}_{ts_name}_{time_str}_{dr_str}"
                                        out_path = dataset_res_dir / "Log-NCDE" / f"{exp_name}_seed-{seed}.json"

                                        if not out_path.exists():
                                            model = LogNeuralCDE(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                                log_ncde_params=log_params, tol=tol, 
                                                                add_time=add_time_bool, t_grid=model_t_grid)
                                            
                                            res = run_experiment(model, data_tuple, cfg, logger, exp_name, "log_ncde", log_params,dr=drop_rate)
                                            res.update(base_info)
                                            res.update(log_params)
                                            save_json(out_path, res)

                            # ===============================================
                            # HELPER FOR Q-FORMER & CONV PARAM GRIDS
                            # ===============================================
                            def get_bw_configs(cfg_block):
                                confs = []
                                if 'bandwidth_lists_different' in cfg_block:
                                    confs.extend([('diff', bw) for bw in cfg_block['bandwidth_lists_different']])
                                if 'bandwidth_lists_same' in cfg_block:
                                    confs.extend([('same', bw) for bw in cfg_block['bandwidth_lists_same']])
                                return confs
                            
                            gp_noises = cfg.get('gp_params', {}).get('noise_stds', [0.01])

                            # ===============================================
                            # EXPERIMENT E: Q-FORMER
                            # ===============================================
                            if cfg['experiments_to_run'].get('qformer', False):
                                q_cfg = cfg['qformer_params']
                                bw_configs = get_bw_configs(q_cfg)
                                
                                for aggr in q_cfg.get('aggregations', ['concat']):
                                    for (bw_type, bws_raw) in bw_configs:
                                        bws_scaled = [round(b * bw_multiplier, 4) for b in bws_raw]
                                        bws_str = str(bws_scaled).replace(' ', '')
                                        
                                        for k_name in q_cfg.get('kernels', []):
                                            noises_to_run = gp_noises if k_name == 'gp' else [None]
                                            
                                            for noise in noises_to_run:
                                                q_params = {
                                                    'kernel': k_name, 'bandwidths': bws_scaled, 
                                                    'tol': tol, 'aggregation': aggr
                                                }
                                                noise_suffix = ""
                                                if noise is not None:
                                                    q_params['noise_std'] = noise
                                                    noise_suffix = f"_noise-{noise}"
                                                
                                                exp_name = f"qformer_{k_name}_{bw_type}_ls-{bws_str}{noise_suffix}_tol-{tol}_aggr-{aggr}_{ts_name}_{time_str}_{reg_str}_{dr_str}"
                                                folder = "gp" if k_name == 'gp' else k_name
                                                out_path = dataset_res_dir / "qformer" / bw_type / folder / f"{exp_name}_seed-{seed}.json"
                                                
                                                if not out_path.exists():
                                                    model = QFormerCDE(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                                    qformer_params=q_params, add_time=add_time_bool, t_grid=model_t_grid)
                                                    
                                                    res = run_experiment(model, data_tuple, cfg, logger, exp_name, "qformer", q_params,dr=drop_rate)
                                                    res.update(base_info)
                                                    res.update(q_params)
                                                    res['bws_raw'] = bws_raw
                                                    save_json(out_path, res)

                            # ===============================================
                            # EXPERIMENT F: CONV
                            # ===============================================
                            if cfg['experiments_to_run'].get('conv', False):
                                c_cfg = cfg['conv_params']
                                k_size = c_cfg['kernel_size']
                                bw_configs = get_bw_configs(c_cfg)
                                kernels_list = c_cfg.get('kernels', ['laplacian']) 

                                for aggr in c_cfg.get('aggregations', ['concat']):
                                    for (bw_type, bws_raw) in bw_configs:
                                        bws_scaled = [round(b * bw_multiplier, 4) for b in bws_raw]
                                        bws_str = str(bws_scaled).replace(' ', '')
                                        
                                        for k_name in kernels_list:
                                            noises_to_run = gp_noises if k_name == 'gp' else [None]
                                            
                                            for noise in noises_to_run:
                                                c_params = {
                                                    'type': 'conv', 'kernel': k_name, 
                                                    'bandwidths': bws_scaled, 'tol': tol, 
                                                    'aggregation': aggr, 'conv_kernel_size': k_size
                                                }
                                                noise_suffix = ""
                                                if noise is not None:
                                                    c_params['noise_std'] = noise
                                                    noise_suffix = f"_noise-{noise}"
                                                
                                                exp_name = f"conv_{k_name}_{bw_type}_ls-{bws_str}{noise_suffix}_tol-{tol}_aggr-{aggr}_ks-{k_size}_{ts_name}_{time_str}_{reg_str}_{dr_str}"
                                                folder = "gp" if k_name == 'gp' else k_name
                                                out_path = dataset_res_dir / "conv" / bw_type / folder / f"{exp_name}_seed-{seed}.json"

                                                if not out_path.exists():
                                                    model = ConvCDE(input_channels, cfg['hidden_dim'], output_channels, seq_len, 
                                                                    conv_params=c_params, add_time=add_time_bool, t_grid=model_t_grid)
                                                    
                                                    res = run_experiment(model, data_tuple, cfg, logger, exp_name, "conv", c_params,dr=drop_rate)
                                                    res.update(base_info)
                                                    res.update(c_params)
                                                    res['bws_raw'] = bws_raw
                                                    save_json(out_path, res)

    logger.info("=== ALL EXPERIMENTS COMPLETED ===")

if __name__ == "__main__":
    main()