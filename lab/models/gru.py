import torch
import torch.nn as nn


class PriceGenGRU(nn.Module):
    """
    Tiny GRU-based model for 15s candle generation.

    Input:  [batch, seq_len, input_dim]   (64 x 17 in your current dataset)
    Output: [batch, horizon, 5]           (OHLCV for next horizon candles)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # For now horizon=1, we predict one next candle from the final hidden state
        self.fc = nn.Linear(hidden_dim, 5)  # OHLCV

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        returns: [batch, horizon, 5]
        """
        out, h_n = self.gru(x)  # out: [batch, seq_len, hidden_dim]
        last_hidden = out[:, -1, :]  # [batch, hidden_dim]
        pred = self.fc(last_hidden)  # [batch, 5]

        # Expand to [batch, horizon, 5] even if horizon=1, to keep shape consistent
        pred = pred.unsqueeze(1).expand(-1, self.horizon, -1).contiguous()
        return pred
