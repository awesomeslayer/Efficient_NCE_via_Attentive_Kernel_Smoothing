import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from .model.model import ConvCDE, NeuralCDE
from .revin import RevIN


class ForecastingLightningModule(pl.LightningModule):
    """
    Unified forecasting LightningModule using Neural CDE or Conv CDE encoder.

    The module takes a time series, embeds it using a CDE-based model,
    and predicts future values using a simple linear head.

    Architecture:
        1) RevIN normalization
        2) CDE-based encoder (NeuralCDE or ConvCDE) -> embedding
        3) LayerNorm + Dropout
        4) Linear head -> predictions
        5) RevIN denormalization
    """

    def __init__(
        self,
        model_type: str = "mvc",
        channels: int = 1,
        num_timefeats: int = 0,
        hidden_channels: int = 64,
        context_len: int = 48,
        horizon_len: int = 24,
        interp_type: str = "gp",
        lr: float = 1e-3,
        tol: float = 1e-4,
        bandwidth: float = 1.0,
        n_heads: int = 4,
    ):
        """
        Args:
            model_type: Type of encoder ('mvc', 'cde', or 'gru')
            channels: Number of input features
            num_timefeats: Number of time features
            hidden_channels: Hidden dimension for the model
            context_len: Length of context/history window
            horizon_len: Length of target/horizon to predict
            interp_type: Interpolation type for NeuralCDE ('cubic' or 'linear')
            lr: Learning rate
            tol: Tolerance for CDE integration
            bandwidth: Bandwidth for ConvCDE kernel
            n_heads: Number of attention heads for ConvCDE
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.channels = channels
        self.num_timefeats = num_timefeats
        self.hidden_channels = hidden_channels
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.interp_type = interp_type
        self.lr = lr
        self.tol = tol

        t_grid = torch.linspace(0, context_len - 1, context_len)
        input_channels = channels + num_timefeats

        if model_type == "gru":
            self.base_model = GRUModel(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=hidden_channels,
            )
        elif model_type == "mvc":
            conv_params = {
                "kernel": interp_type,
                "bandwidths": [bandwidth] * n_heads,
                "aggregation": "concat",
                "conv_kernel_size": 3,
                "tol": tol,
            }
            self.base_model = ConvCDE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=hidden_channels,
                seq_len=context_len,
                conv_params=conv_params,
                add_time=True,
                t_grid=t_grid,
            )
        elif model_type == "cde":
            self.base_model = NeuralCDE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=hidden_channels,
                seq_len=context_len,
                interpolation=interp_type,
                add_time=True,
                t_grid=t_grid,
            )
        elif model_type == "lin":
            self.base_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    context_len * (channels + num_timefeats), horizon_len * channels
                ),
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.revin = RevIN(num_features=channels)

        if model_type != "lin":
            # Prediction head and regularization
            self.head = nn.Linear(hidden_channels, channels * horizon_len)
        else:
            self.head = nn.Identity()

    def forward(self, x, t):
        """
        Predict future values from a sequence.

        Args:
            x: Input sequence of shape (batch, context_len, channels)
            t: Time features of shape (batch, context_len, num_timefeats)

        Returns:
            preds: Predicted future values (batch, horizon_len, channels)
        """
        # RevIN normalization
        x = self.revin(x, mode="norm")

        # Concatenate data with time features
        inp = torch.cat([x, t], dim=-1)

        # Encode the input sequence to embedding
        emb = self.base_model(inp)

        # Apply linear prediction head
        preds = self.head(emb).reshape(-1, self.horizon_len, self.channels)

        # RevIN denormalization
        preds = self.revin(preds, mode="denorm")
        return preds

    def training_step(self, batch, batch_idx):
        """Training step with MSE loss."""
        x, y, t = batch
        preds = self.forward(x, t)
        loss = F.mse_loss(preds, y)
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with MSE."""
        x, y, t = batch
        preds = self.forward(x, t)
        mse = F.mse_loss(preds, y)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step with MSE."""
        x, y, t = batch
        preds = self.forward(x, t)
        mse = F.mse_loss(preds, y)
        self.log("test_mse", mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
