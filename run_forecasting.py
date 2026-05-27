import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from forecasting.datamodule import ForecastingDataModule
from forecasting.forecasting_module import ForecastingLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Forecasting with Neural CDEs")

    # Data arguments
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file with forecasting data (e.g., ETTh1.csv)",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=96,
        help="Length of the history/context patch",
    )
    parser.add_argument(
        "--horizon_len", type=int, default=96, help="Length of the target/future patch"
    )
    parser.add_argument(
        "--irregularity_fraction",
        type=float,
        default=0.7,
        help="Fraction of time points to keep (1.0 = no missing, 0.7 = 30%% missing)",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="mvc",
        choices=["mvc", "cde", "gru", "lin"],
        help="Model type: 'mvc' (ConvCDE), 'cde' (NeuralCDE), or 'gru' (GRU)",
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Hidden dimension for the model"
    )
    parser.add_argument(
        "--bandwidth", type=float, default=8.0, help="Bandwidth for ConvCDE kernel"
    )
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Number of heads for ConvCDE"
    )
    parser.add_argument(
        "--interp_type",
        type=str,
        default="cubic",
        choices=["cubic", "linear", "gaussian", "gp"],
        help="Interpolation type",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3, help="Tolerance for CDE solver"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoaders"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for DataLoaders"
    )

    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="forecasting",
        help="MLFlow experiment name",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices (GPUs) to use"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, gpu, cpu)",
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Initialize MLFlow Logger (before seeding to make different names)
    mlflow_logger = MLFlowLogger(
        experiment_name=args.mlflow_experiment_name,
        log_model=True,
        tracking_uri="sqlite:///mlflow.db",
    )

    # Log hyperparameters
    mlflow_logger.log_hyperparams(
        {
            "model_type": args.model_type,
            "hidden_channels": args.hidden_channels,
            "interp_type": args.interp_type,
            "bandwidth": args.bandwidth,
            "n_heads": args.n_heads,
            "context_len": args.context_len,
            "horizon_len": args.horizon_len,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "tol": args.tol,
            "csv_path": args.csv_path,
            "seed": args.seed,
            "irregularity_fraction": args.irregularity_fraction,
        }
    )

    # Set seed
    set_seed(args.seed)

    # Initialize DataModule
    print(f"Loading data from: {args.csv_path}")
    datamodule = ForecastingDataModule(
        csv_path=args.csv_path,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        irregularity_fraction=args.irregularity_fraction,
    )
    datamodule.setup()

    # Get feature dimensions
    channels = datamodule.get_feature_dim()
    num_timefeats = datamodule.get_num_timefeats()

    print(f"Data loaded:")
    print(f"  Train: {len(datamodule.train_dataset)} patches")
    print(f"  Val:   {len(datamodule.val_dataset)} patches")
    print(f"  Test:  {len(datamodule.test_dataset)} patches")
    print(f"  Channels: {channels}, Time features: {num_timefeats}")

    # Initialize the unified ForecastingLightningModule
    model = ForecastingLightningModule(
        model_type=args.model_type,
        channels=channels,
        num_timefeats=num_timefeats,
        hidden_channels=args.hidden_channels,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        interp_type=args.interp_type,
        lr=args.lr,
        tol=args.tol,
        bandwidth=args.bandwidth,
        n_heads=args.n_heads,
    )

    print(f"\nModel: ForecastingLightningModule ({args.model_type.upper()}CDE)")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Interp type: {args.interp_type}")
    print(f"  Context len: {args.context_len}")
    print(f"  Horizon len: {args.horizon_len}")

    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor="val_mse", mode="min")

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        logger=mlflow_logger,
        enable_checkpointing=True,
        default_root_dir="./forecasting_checkpoints",
        callbacks=[ckpt_callback],
    )

    # Train the model
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.fit(model, datamodule)

    # Test the model
    print("\n" + "=" * 50)
    print("Starting testing...")
    print("=" * 50 + "\n")

    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
