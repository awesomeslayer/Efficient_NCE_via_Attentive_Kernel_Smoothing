import math

import torch
import torch.nn as nn
import torchcde
from torch.nn import functional as F

from .interpolation import WeightedGPInterpolation, WeightedKernelInterpolation


class BaseModel(nn.Module):
    def __init__(self, t_grid=None):
        super().__init__()
        if t_grid is not None:
            self.register_buffer("t_grid", t_grid)


class SimpleCDEFunc(nn.Module):
    """Simple CDE function: dz/dt = f(t, z)."""

    def __init__(self, input_channels, hidden_channels, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels * input_channels),
            nn.Tanh(),
        )
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

    def forward(self, t, z):
        return self.net(z).reshape(-1, self.hidden_channels, self.input_channels)


class NeuralCDE(BaseModel):
    """Neural CDE model that encodes time series into a hidden state."""

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        seq_len,
        interpolation="cubic",
        tol=1e-3,
        add_time=True,
        t_grid=None,
    ):
        super().__init__(t_grid)
        self.cde_func = SimpleCDEFunc(input_channels, hidden_channels, seq_len)
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation
        self.tol = tol
        self.add_time = add_time

    def make_interpolation(self, coeffs):
        t = self.t_grid if self.t_grid is not None else None

        if self.interpolation == "cubic":
            return torchcde.CubicSpline(coeffs, t=t)
        if self.interpolation == "linear":
            return torchcde.LinearInterpolation(coeffs, t=t)
        raise ValueError(f"Unknown interpolation: {self.interpolation}")

    def forward(self, x):
        """
        Encode time series to embedding.

        Args:
            x: Input of shape (batch, seq_len, input_channels) - raw data (coefficients are computed internally)

        Returns:
            embedding: Hidden state at end of sequence (batch, output_channels)
        """
        coeffs = torchcde.natural_cubic_coeffs(x)
        X = self.make_interpolation(coeffs)

        z0 = self.initial(X.evaluate(X.interval[0]))

        z_T = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z0,
            t=X.interval,
            atol=self.tol,
            rtol=self.tol,
            adjoint=False,
        )

        return self.readout(z_T[:, 1])


class ParallelCDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads, seq_len):
        super().__init__()
        self.input_channels, self.hidden_channels, self.num_heads = (
            input_channels,
            hidden_channels,
            num_heads,
        )

        self.w1 = nn.Parameter(torch.empty(num_heads, hidden_channels, hidden_channels))
        self.b1 = nn.Parameter(torch.empty(num_heads, hidden_channels))
        self.w2 = nn.Parameter(
            torch.empty(num_heads, hidden_channels * input_channels, hidden_channels)
        )
        self.b2 = nn.Parameter(torch.empty(num_heads, hidden_channels * input_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.w1, self.w2]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.hidden_channels)
        for b in [self.b1, self.b2]:
            nn.init.uniform_(b, -bound, bound)

    def forward(self, t, z):
        batch_size = z.size(0) // self.num_heads
        z_in = z.view(batch_size, self.num_heads, self.hidden_channels)

        h = F.relu(torch.einsum("hoi,bhi->bho", self.w1, z_in) + self.b1)

        out = torch.tanh(torch.einsum("hoi,bhi->bho", self.w2, h) + self.b2)

        return out.reshape(z.size(0), self.hidden_channels, self.input_channels)


class MultiHeadCDEBase(BaseModel):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        seq_len,
        params,
        add_time,
        t_grid,
    ):
        super().__init__(t_grid)
        p = params
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel = p.get("kernel", "gaussian")
        self.bandwidths = p.get("bandwidths", [1.0])
        self.noise_std = p.get("noise_std", 0.01)
        self.tol = p.get("tol", 1e-4)
        self.aggregation = p.get("aggregation", "concat")
        self.add_time = add_time

        self.num_heads = len(self.bandwidths)

        self.cde_func = ParallelCDEFunc(
            input_channels, hidden_channels, self.num_heads, seq_len
        )

        self.initial_layer = nn.Linear(input_channels, hidden_channels)

        if self.aggregation == "concat":
            r_dim = hidden_channels * self.num_heads
        else:
            r_dim = hidden_channels

        self.readout = nn.Linear(r_dim, output_channels)

    def _get_weighted_interp(self, coeffs, weights):
        batch_size = coeffs.size(0)

        c_par = coeffs.repeat_interleave(self.num_heads, dim=0)

        w_par = weights.reshape(-1, weights.size(-1))

        b_tensor = torch.tensor(
            self.bandwidths, device=coeffs.device, dtype=coeffs.dtype
        ).repeat(batch_size)

        if self.kernel == "gp":
            return WeightedGPInterpolation(
                c_par,
                w_par,
                self.t_grid,
                {"length_scale": b_tensor, "noise_std": self.noise_std},
                self.add_time,
            )

        return WeightedKernelInterpolation(
            c_par,
            w_par,
            self.t_grid,
            {"kernel": self.kernel, "bandwidth": b_tensor},
            self.add_time,
        )

    def _aggregate(self, z_T, batch_size):
        final_flat = z_T[:, 1, :]

        final = final_flat.view(batch_size, self.num_heads, self.hidden_channels)

        if self.aggregation == "concat":
            return final.reshape(batch_size, -1)
        if self.aggregation == "mean":
            return final.mean(dim=1)
        if self.aggregation == "max":
            return final.max(dim=1)[0]
        return final.reshape(batch_size, -1)


class QFormerCDE(MultiHeadCDEBase):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        seq_len,
        qformer_params={},
        add_time=True,
        t_grid=None,
    ):
        super().__init__(
            input_channels,
            hidden_channels,
            output_channels,
            seq_len,
            qformer_params,
            add_time,
            t_grid,
        )

        raw_dim = input_channels
        self._last_attn = None
        self.queries = nn.Parameter(torch.randn(self.num_heads, raw_dim))

    def forward(self, X):
        self.cde_func.nfe = 0

        attn_scores = torch.einsum("md,bld->bml", self.queries, X) / math.sqrt(
            self.queries.size(-1)
        )

        start_feat = 1 if self.add_time else 0
        missing_mask = torch.isnan(X[..., start_feat]).unsqueeze(1)
        attn_scores = attn_scores.masked_fill(missing_mask, -1e9)

        attn = F.softmax(attn_scores, dim=-1)

        if not self.training:
            self._last_attn = attn.detach().cpu()

        X = self._get_weighted_interp(X, attn)

        z0 = self.initial_layer(X.evaluate(X.interval[0]))

        z_T = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z0,
            t=X.interval,
            atol=self.tol,
            rtol=self.tol,
            adjoint=False,
        )

        return self.readout(self._aggregate(z_T, X.size(0)))


class ConvCDE(MultiHeadCDEBase):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        seq_len,
        conv_params={},
        add_time=True,
        t_grid=None,
    ):
        super().__init__(
            input_channels,
            hidden_channels,
            output_channels,
            seq_len,
            conv_params,
            add_time,
            t_grid,
        )

        self._last_attn = None

        ks = conv_params.get("conv_kernel_size", 3)
        self.net = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_channels,
                ks,
                padding=ks // 2,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                ks,
                padding=ks // 2,
                padding_mode="replicate",
            ),
            nn.ReLU(),
        )
        self.to_heads = nn.Linear(hidden_channels, self.num_heads)

    def forward(self, x):
        weights = self.net(x.transpose(1, -1)).transpose(1, -1)

        attn_scores = self.to_heads(weights).transpose(1, 2)

        start_feat = 1 if self.add_time else 0
        missing_mask = torch.isnan(x[..., start_feat]).unsqueeze(1)
        attn_scores = attn_scores.masked_fill(missing_mask, -1e9)

        attn = F.softmax(attn_scores, dim=-1)

        if not self.training:
            self._last_attn = attn.detach().cpu()

        X = self._get_weighted_interp(x, attn)

        z0 = self.initial_layer(X.evaluate(X.interval[0]))

        z_T = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z0,
            t=X.interval,
            atol=self.tol,
            rtol=self.tol,
            adjoint=False,
        )

        return self.readout(self._aggregate(z_T, x.size(0)))


class GRUModel(nn.Module):
    """Simple GRU model that encodes time series into a hidden state."""

    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.gru = nn.GRU(
            input_channels,
            hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, x):
        """
        Encode time series to embedding using GRU.

        Args:
            x: Input of shape (batch, seq_len, input_channels)

        Returns:
            embedding: Hidden state at end of sequence (batch, output_channels)
        """
        output, _ = self.gru(x)
        return self.readout(output[:, -1, :])
