import torch
import torch.nn as nn

"edit MKR (LSTM-AE -> BiLSTM-AE)"
# 단변량 및 다변량 공통
# 입출력 길이 같음

# https://arxiv.org/pdf/1607.00148.pdf
class BiLSTMAutoEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0,
                 device=torch.device('cpu'), merge="linear"):
        """
        merge:
          - "sum":   forward+backward hidden을 더해서 decoder에 전달
          - "linear": concat 후 선형변환으로 hidden_size로 맞춤 (권장)
        """
        super().__init__()
        self.device = device
        self.encoder = BiEncoder(num_layers, hidden_size, nb_feature, dropout, device, merge=merge)
        self.decoder = Decoder(num_layers, hidden_size, nb_feature, dropout, device)

    def forward(self, input_seq):
        # (B, L, D)
        output = torch.zeros_like(input_seq, dtype=torch.float, device=input_seq.device)

        hidden_cell = self.encoder(input_seq)  # (h0, c0) each: (num_layers, B, hidden_size)

        # 마지막 시점 feature 벡터를 decoder 첫 입력으로 사용 (B, 1, D)
        input_decoder = input_seq[:, -1, :].unsqueeze(1)

        # 역순 재구성
        for i in range(input_seq.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output[:, i, :] = output_decoder[:, 0, :]

        return output


class BiEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0,
                 device=torch.device('cpu'), merge="linear"):
        super().__init__()
        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.merge = merge

        self.lstm = nn.LSTM(
            input_size=nb_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bias=True,
            bidirectional=True
        )

        # concat(2*hidden) -> hidden 로 투영해서 decoder와 차원 맞춤
        if self.merge == "linear":
            self.h_proj = nn.Linear(2 * hidden_size, hidden_size)
            self.c_proj = nn.Linear(2 * hidden_size, hidden_size)
        elif self.merge == "sum":
            self.h_proj = None
            self.c_proj = None
        else:
            raise ValueError("merge must be 'linear' or 'sum'")

    def initHidden(self, batch_size, device):
        # BiLSTM는 num_layers*2
        h0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size), dtype=torch.float, device=device)
        c0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size), dtype=torch.float, device=device)
        return (h0, c0)

    def _merge_directions(self, x):
        """
        x: (num_layers*2, B, H)  ->  (num_layers, B, H)
        """
        # (num_layers, 2, B, H)
        x = x.view(self.num_layers, 2, x.size(1), x.size(2))

        fwd = x[:, 0, :, :]  # (num_layers, B, H)
        bwd = x[:, 1, :, :]  # (num_layers, B, H)

        if self.merge == "sum":
            return fwd + bwd  # (num_layers, B, H)

        # concat -> linear
        cat = torch.cat([fwd, bwd], dim=-1)  # (num_layers, B, 2H)
        return cat  # projection은 밖에서

    def forward(self, input_seq):
        # input_seq: (B, L, D)
        batch_size = input_seq.size(0)
        device = input_seq.device

        h0, c0 = self.initHidden(batch_size, device)
        _, (h_n, c_n) = self.lstm(input_seq, (h0, c0))  # each: (num_layers*2, B, H)

        h = self._merge_directions(h_n)  # (num_layers, B, H) or (num_layers, B, 2H)
        c = self._merge_directions(c_n)

        if self.merge == "linear":
            # (num_layers, B, 2H) -> (num_layers, B, H)
            h = self.h_proj(h)
            c = self.c_proj(c)

        return (h, c)


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super().__init__()
        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=nb_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bias=True
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)

    def forward(self, input_seq, hidden_cell):
        output, hidden_cell = self.lstm(input_seq, hidden_cell)  # output: (B, 1, H)
        output = self.linear(output)  # (B, 1, D)
        return output, hidden_cell