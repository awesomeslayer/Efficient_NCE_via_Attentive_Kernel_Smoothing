import os
import pickle
import torch
import numpy as np
import torchaudio
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_classification
import urllib.request
import tarfile
import tempfile
import shutil
import pandas as pd
import zipfile
import io
import warnings
from torch.utils.data import DataLoader

def _load_and_process_speech_commands(save_path, logger):
    logger.info("Downloading and processing SpeechCommands dataset... This will take a few minutes.")
    
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=os.path.dirname(save_path), download=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using {device} for fast MFCC extraction...")

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 100, "n_mels": 23, "center": False}
    ).to(device)
    
    def collate_fn(batch):
        waveforms = []
        labels = []
        for item in batch:
            waveform = item[0] # (1, time)
            if waveform.shape[1] < 16000:
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
            elif waveform.shape[1] > 16000:
                waveform = waveform[:, :16000]
            waveforms.append(waveform)
            labels.append(item[2])
        return torch.cat(waveforms, dim=0), labels

    loader = DataLoader(
        dataset, 
        batch_size=1024, 
        collate_fn=collate_fn, 
        num_workers=4, 
        shuffle=False
    )
    
    X_list = []
    y_list = []
    
    logger.info("Extracting MFCC features from 105,000+ audio files in batches...")
    
    for waveforms, labels in loader:
        waveforms = waveforms.to(device) 
        
        with torch.no_grad():
            mfcc = mfcc_transform(waveforms) 
            
        X_list.append(mfcc.cpu().numpy())
        y_list.extend(labels)
        
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
        
    logger.info(f"Speech Commands processed and cached. Final shape: {X.shape}")
    return X, y



def _load_and_process_sepsis_2019(save_path, logger):
    logger.info("Processing Sepsis-2019 from single Kaggle CSV...")
    
    csv_path = "data/Dataset.csv" #
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    
    if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])

    logger.info(f"Loaded {len(df)} rows. Grouping by Patient_ID...")

    X_list = []
    y_list = []

    grouped = df.groupby('Patient_ID')
    
    count = 0
    for patient_id, group in grouped:
        group = group.sort_values('Hour')
        
        group = group.iloc[:72]
        
        label = 1 if group['SepsisLabel'].max() > 0 else 0
        
        features = group.drop(columns=['SepsisLabel', 'Patient_ID']).values
        
        if len(features) >= 2:
            X_list.append(features.astype(np.float32))
            y_list.append(label)
        
        count += 1
        if count % 5000 == 0:
            logger.info(f"Processed {count} patients...")

    max_len = 72
    num_features = X_list[0].shape[1]
    X_padded = np.full((len(X_list), max_len, num_features), np.nan, dtype=np.float32)
    
    for i, x in enumerate(X_list):
        curr_len = x.shape[0]
        X_padded[i, :curr_len, :] = x
        
    X = X_padded
    y = np.array(y_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
        
    logger.info(f"Sepsis-2019 ready. Patients: {len(X)}, Features: {num_features}")
    return X, y

def apply_drop_rate(X, drop_rate, add_time_str):
    if drop_rate <= 0.0:
        return X.clone()
    
    X_dropped = X.clone()
    batch_size, seq_len, num_features = X.shape
    
    valid_indices = torch.arange(1, seq_len - 1)
    num_to_drop = int(drop_rate * len(valid_indices))
    
    start_feat_idx = 1 if str(add_time_str).lower() == 'yes' else 0
    
    for b in range(batch_size):
        drop_idx = valid_indices[torch.randperm(len(valid_indices))[:num_to_drop]]
        X_dropped[b, drop_idx, start_feat_idx:] = float('nan')
        
    return X_dropped

def get_data(cfg, logger):
    dataset_name = cfg['dataset_name']
    save_path = os.path.join(cfg['data_dir'], f"{dataset_name}.pkl")

    logger.info(f"--- Loading dataset: {dataset_name} ---")
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        X, y = data['X'], data['y']
    else:
        if dataset_name.lower() == 'speechcommands':
            X, y = _load_and_process_speech_commands(save_path, logger)
        elif dataset_name.lower() == 'sepsis2019':
            X, y = _load_and_process_sepsis_2019(save_path, logger)
        else:
            X, y, _ = load_classification(dataset_name, return_metadata=True)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)
            logger.info("Data was downloaded from aeon and saved.")
      

    if not np.issubdtype(y.dtype, np.number):
        logger.info("Non-numeric labels detected, encoding with LabelEncoder.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        y = y.astype(int)
    
    n_classes = len(np.unique(y))
    logger.info(f"Dataset '{dataset_name}' loaded. Shape: {X.shape}, Classes: {n_classes}")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    time_scaling = cfg['time_scaling_factor']
    seq_len = X.shape[2]
    t_original = torch.arange(0, seq_len, dtype=X.dtype)

    if isinstance(time_scaling, str):
        if time_scaling.lower() == 'uni':
            logger.info("Time scaling set to 'uni'. Force creating timestamps linspace(0, 1).")
            timestamps = torch.linspace(0, 1, seq_len, dtype=X.dtype)
            
        elif time_scaling.lower() == 'no':
            logger.info("Time scaling set to 'no'. Using regular timestamps (0, 1, 2...).")
            timestamps = t_original
        else:
            try:
                ts_val = float(time_scaling)
                norm_const = float(seq_len - 1)
                timestamps = (ts_val / norm_const) * t_original
                logger.info(f"Time scaling factor (from str): {ts_val}. Normalized timestamps.")
            except ValueError:
                 raise ValueError(f"Invalid time_scaling_factor: {time_scaling}. Must be 'uni', 'no', or a number.")
    else:
        try:
            ts_val = float(time_scaling)
            norm_const = float(seq_len - 1)
            timestamps = (ts_val / norm_const) * t_original
            logger.info(f"Time scaling factor: {ts_val}. Normalized timestamps.")
        except ValueError:
             raise ValueError(f"Invalid time_scaling_factor: {time_scaling}. Must be a number, 'uni' or 'no'.")
    
    add_time_str = str(cfg.get('add_time', 'yes')).lower()
    
    X = X.permute(0, 2, 1) 

    if add_time_str == 'yes':
        timestamps_expanded = timestamps.unsqueeze(0).unsqueeze(0).expand(X.size(0), 1, -1)
        timestamps_expanded = timestamps_expanded.permute(0, 2, 1) # (batch, seq, 1)
        X = torch.cat([timestamps_expanded, X], dim=2)
        logger.info("Time channel appended to X.")
    else:
        logger.info("Time channel NOT appended to X (kept separate).")

    perm = torch.randperm(X.size(0))
    X, y = X[perm], y[perm]
    logger.info(f"Data prepared. Final X shape: {X.shape}")

    train_frac, val_frac = 0.6, 0.2
    num_samples = X.shape[0]
    train_end = int(num_samples * train_frac)
    val_end = int(num_samples * (train_frac + val_frac))

    train_X, val_X, test_X = X[:train_end], X[train_end:val_end], X[val_end:]
    train_y, val_y, test_y = y[:train_end], y[train_end:val_end], y[val_end:]
    logger.info(f"Train/Val/Test split: {len(train_X)}/{len(val_X)}/{len(test_X)}")

    
    start_feat_idx = 1 if add_time_str == 'yes' else 0
    
    logger.info("Standardizing features based on training set statistics (ignoring NaNs).")
    
    
    for feature_idx in range(start_feat_idx, train_X.shape[2]):
        valid_mask = ~torch.isnan(train_X[:, :, feature_idx])
        valid_data = train_X[:, :, feature_idx][valid_mask]
        
        if len(valid_data) > 0:
            mean = valid_data.mean()
            std = valid_data.std() + 1e-8
            
            train_X[:, :, feature_idx] = (train_X[:, :, feature_idx] - mean) / std
            val_X[:, :, feature_idx] = (val_X[:, :, feature_idx] - mean) / std
            test_X[:, :, feature_idx] = (test_X[:, :, feature_idx] - mean) / std

    logger.info("\n" + "="*40)
    logger.info("      DATA INSPECTION (After Norm)      ")
    logger.info("="*40)
    
    if add_time_str == 'yes':
        ts_snippet = train_X[0, :10, 0]
        ts_min = train_X[:, :, 0].min().item()
        ts_max = train_X[:, :, 0].max().item()
        feat_snippet = train_X[0, :5, 1:]
        all_feats = train_X[:, :, 1:]
        logger.info(f"TIMESTAMPS (Sample 0, first 10 steps, embedded):\n{ts_snippet}")
        logger.info(f"TIMESTAMPS RANGE (Embedded): Min={ts_min:.4f}, Max={ts_max:.4f}")
    else:
        ts_snippet = timestamps[:10]
        ts_min = timestamps.min().item()
        ts_max = timestamps.max().item()
        feat_snippet = train_X[0, :5, :]
        all_feats = train_X
        logger.info(f"TIMESTAMPS (Separate Tensor, first 10 steps):\n{ts_snippet}")
        logger.info(f"TIMESTAMPS RANGE (Separate): Min={ts_min:.4f}, Max={ts_max:.4f}")

    logger.info(f"FEATURES (Sample 0, first 5 steps):\n{feat_snippet}")
    logger.info(f"FEATURES STATS (Global Train): Mean={all_feats.mean():.4f}, Std={all_feats.std():.4f}")

    logger.info(f"LABELS (First 15): {train_y[:15].tolist()}")
    logger.info("="*40 + "\n")
    
    return train_X, val_X, test_X, train_y, val_y, test_y, timestamps