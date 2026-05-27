from pathlib import Path

import polars as pl
import torch
from pandas.core.arrays.datetimes import np
from polars import selectors as cs
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .timefeats import extract_time_feats


class PatchDataset(Dataset):
    """
    Dataset that returns (context, target) patches with synthetic irregularity.

    Each sample consists of:
        - x: context patch (history) of length context_len, indexed by a patch of irregular indices
        - y: target patch (future) of length horizon_len, taken from original data after max x index
        - t: time features of length context_len + horizon_len, indexed by the same irregular indices

    Irregularity is simulated by selecting a fraction of time points and indexing into them.
    """

    def __init__(
        self,
        variates: torch.Tensor,
        timefeats: torch.Tensor,
        indices: torch.Tensor,
        context_len: int,
        horizon_len: int,
    ):
        """
        Args:
            variates: Full data tensor of shape (total_length, num_features)
            timefeats: Time features tensor of shape (total_length, num_timefeats)
            indices: Selected indices representing kept/irregular time points
            context_len: Length of the history/context patch
            horizon_len: Length of the target/future patch
        """
        self.variates = variates
        self.timefeats = timefeats
        self.indices = indices
        self.context_len = context_len
        self.horizon_len = horizon_len

        # Number of possible patches given irregular indices
        # Need at least context_len indices for x, plus horizon_len positions for y after max x index
        self.num_patches = len(indices) - context_len - horizon_len + 1

    def __len__(self):
        return max(0, self.num_patches)

    def __getitem__(self, idx):
        """
        Returns:
            x: Context patch of shape (context_len, num_features)
            y: Target patch of shape (horizon_len, num_features)
            t: Time features of shape (context_len + horizon_len, num_timefeats)
        """
        # Get a patch of indices for x
        idx_indices = self.indices[idx : idx + self.context_len]

        # Extract x using these irregular indices
        x = self.variates[idx_indices]

        # For y: find the last x index, then take horizon_len positions after it from full data
        last_x_idx = idx_indices[-1].item()
        y_start = last_x_idx + 1
        y_end = y_start + self.horizon_len
        y = self.variates[y_start:y_end]

        # Time features for context
        t = self.timefeats[idx_indices]

        return x, y, t


class ForecastingDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for time series forecasting.

    Loads CSV data, splits into train/val/test (consecutive, no shuffling),
    and creates patch-based datasets for autoregressive next-value forecasting.

    Supports synthetic irregularity via irregularity_fraction parameter.
    """

    def __init__(
        self,
        csv_path: str,
        context_len: int = 96,
        horizon_len: int = 96,
        batch_size: int = 32,
        num_workers: int = 16,
        irregularity_fraction: float = 1.0,
    ):
        """
        Args:
            csv_path: Path to the .csv file with forecasting data
            context_len: Length of the history/context patch (x)
            horizon_len: Length of the target/future patch (y)
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
        """
        super().__init__()
        self.save_hyperparameters()

        self.csv_path = Path(csv_path)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.irregularity_fraction = irregularity_fraction

        # Data tensors (raw normalized values)
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Time features (raw)
        self.train_timefeats = None
        self.val_timefeats = None
        self.test_timefeats = None

        # Selected indices for synthetic irregularity
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Scaler for normalization
        self.scaler = None
        self.num_variates = None
        self.num_timefeats = None

    def load_data(self):
        """Load data from CSV file."""
        df = pl.read_csv(self.csv_path, try_parse_dates=True)
        freq = df.select(cs.temporal().diff().mode()).item().seconds
        daylen = 24 * 60 * 60 // freq
        df = df[: df.height - df.height % daylen]  # take full days
        data = df.select(cs.numeric()).to_numpy()
        timefeats = extract_time_feats("date", df)

        all_indices = np.arange(data.shape[0])
        indices_daily = all_indices.reshape(-1, daylen)
        rng = np.random.default_rng(seed=42)
        daily_indices_keep = rng.choice(
            indices_daily.shape[0],
            size=int(self.irregularity_fraction * indices_daily.shape[0]),
            replace=False,
        )
        indices_keep = indices_daily[daily_indices_keep].reshape(-1)
        indices_keep = np.sort(indices_keep)

        self.num_variates = data.shape[-1]
        self.num_timefeats = timefeats.shape[1]
        return data, timefeats, indices_keep

    def setup(self, stage: str | None = None):
        """
        Load and split data, create datasets.

        Args:
            stage: Optional stage for fit/predict style compatibility
        """
        # Load data from CSV
        data, timefeats, indices_keep = self.load_data()

        if "ETTh" in self.csv_path.name:
            month = 24 * 30
            train_end = month * 12
            val_end = train_end + month * 4
            test_end = val_end + month * 4
        elif "ETTm" in self.csv_path.name:
            month = 24 * 30 * 4
            train_end = month * 12
            val_end = train_end + month * 4
            test_end = val_end + month * 4
        else:
            # Set split indices (consecutive, no shuffling)
            N = data.shape[0]
            train_end = int(0.7 * N)
            val_end = train_end + int(0.1 * N)
            test_end = N

        # Select indices for synthetic irregularity within each split
        # These indices represent "kept" time points (rows) for irregular sampling
        self.train_indices = indices_keep[indices_keep < train_end]
        self.val_indices = (
            indices_keep[(indices_keep >= train_end) & (indices_keep < val_end)]
            - train_end
        )
        self.test_indices = (
            indices_keep[(indices_keep >= val_end) & (indices_keep < test_end)]
            - val_end
        )

        # Split raw data (no overlap between splits)
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:test_end]

        # Split time features (no overlap)
        train_timefeats = timefeats[:train_end]
        val_timefeats = timefeats[train_end:val_end]
        test_timefeats = timefeats[val_end:test_end]

        # Fit scaler on training data only
        self.scaler = StandardScaler()
        train_data = self.scaler.fit_transform(train_data)
        val_data = self.scaler.transform(val_data)
        test_data = self.scaler.transform(test_data)

        # Convert to tensors
        self.train_data = torch.from_numpy(train_data).float()
        self.val_data = torch.from_numpy(val_data).float()
        self.test_data = torch.from_numpy(test_data).float()

        self.train_timefeats = torch.from_numpy(train_timefeats).float()
        self.val_timefeats = torch.from_numpy(val_timefeats).float()
        self.test_timefeats = torch.from_numpy(test_timefeats).float()

        # Create datasets
        self.train_dataset = self._build_dataset("train")
        self.val_dataset = self._build_dataset("val")
        self.test_dataset = self._build_dataset("test")

    def _build_dataset(self, split: str):
        """Build a PatchDataset for the given split."""
        if split == "train":
            variates = self.train_data
            timefeats = self.train_timefeats
            indices = self.train_indices
        elif split == "val":
            variates = self.val_data
            timefeats = self.val_timefeats
            indices = self.val_indices
        elif split == "test":
            variates = self.test_data
            timefeats = self.test_timefeats
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split: {split}")

        return PatchDataset(
            variates=variates,
            timefeats=timefeats,
            indices=indices,
            context_len=self.context_len,
            horizon_len=self.horizon_len,
        )

    def train_dataloader(self):
        """Return DataLoader for training set."""
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return DataLoader for validation set."""
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return DataLoader for test set."""
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_feature_dim(self):
        """Return the number of features in the data."""
        if self.train_data is None:
            self.setup()
        return self.num_variates

    def get_num_timefeats(self):
        """Return the number of time features."""
        if self.train_data is None:
            self.setup()
        return self.num_timefeats

    def inverse_transform(self, data):
        """Inverse transform normalized data back to original scale."""
        if self.scaler is None:
            raise RuntimeError("Scaler not initialized. Call setup() first.")
        return self.scaler.inverse_transform(data.cpu().numpy())
