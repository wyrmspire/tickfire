import torch
import torch.nn as nn


class PriceGenGRU(nn.Module):
    """
    GRU-based model for sequence-to-next-step prediction.

    Input:
        x: [batch, seq_len, input_dim]
           (e.g. 256 x feat_dim window of 15s candles + indicators)

    Output:
        returns: [batch, horizon, 5]

        The 5 outputs correspond to RETURNS defined as:

          0: (O_{t+h} - C_t) / C_t
          1: (H_{t+h} - C_t) / C_t
          2: (L_{t+h} - C_t) / C_t
          3: (C_{t+h} - C_t) / C_t
          4: (V_{t+h} - V_t) / (V_t + 1)

        where C_t and V_t are the current close and volume.
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
        # Predict 5 return components from final hidden state
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        returns: [batch, horizon, 5]
        """
        out, _ = self.gru(x)              # [batch, seq_len, hidden_dim]
        last_hidden = out[:, -1, :]       # [batch, hidden_dim]
        pred = self.fc(last_hidden)       # [batch, 5]
        pred = pred.unsqueeze(1).expand(-1, self.horizon, -1).contiguous()
        return pred
