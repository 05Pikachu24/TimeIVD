import torch
import torch.nn as nn
import math
from torch_dct import dct, idct
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)

class Patch_Diverse_V1(nn.Module):
    def __init__(self, dim, emb_dim, num_var, split_ratio, noise_factor, high_freq_emphasis):
        super().__init__()
        self.split_ratio = split_ratio
        self.num_var = num_var
        self.d_model = dim
        self.emd_dim = emb_dim
        self.noise_factor = noise_factor
        self.high_freq_emphasis = high_freq_emphasis

        self.low_factor = nn.Parameter(torch.randn(num_var, int(self.emd_dim * split_ratio)))
        self.high_factor = nn.Parameter(torch.randn(num_var, self.emd_dim - int(self.emd_dim * split_ratio)))

        self.uplinear = nn.Linear(self.d_model, self.emd_dim)

    def manipulate_frequency_bands(self, freq_x, low_freq_factor=0.5, high_freq_factor=1.5, split_ratio=0.5):
        seq_len = freq_x.shape[-1]
        split_point = int(seq_len * split_ratio)

        low_freq_components = freq_x[..., :split_point]
        high_freq_components = freq_x[..., split_point:]

        perturbed_low_freq = low_freq_components * low_freq_factor
        perturbed_high_freq = high_freq_components * high_freq_factor

        perturbed_freq_x = torch.cat((perturbed_low_freq, perturbed_high_freq), dim=-1)
        return perturbed_freq_x

    def frequency_dropout(self, freq_x, p=0.2, zero_low_freq=False):

        if not self.training:
            return freq_x

        mask = torch.full_like(freq_x, 1 - p)
        mask = torch.bernoulli(mask)

        if not zero_low_freq:
            mask[..., 0] = 1

        return freq_x * mask if p < 1.0 else freq_x * mask

    def frequency_noise_injection(self, freq_x, noise_factor=0.1, high_freq_emphasis=True):
        # [batch_size, num_features, seq_len]
        noise = torch.randn_like(freq_x)

        if high_freq_emphasis:
            seq_len = freq_x.shape[-1]

            emphasis = torch.logspace(-1, 0, seq_len, device=freq_x.device).view(1, 1, seq_len)

            noise = noise * emphasis

        return freq_x + noise_factor * noise

    def forward(self, x):
        """
        input: [batch_size, num_features, seq_len]
        output: [batch_size, num_features, d_model]
        """
        B, N, D = x.shape
        x = x.to(torch.float32)
        dct_data = dct(x, norm='ortho')
        dct_data = self.uplinear(dct_data)
        dct_data_1 = self.manipulate_frequency_bands(dct_data, low_freq_factor=F.softplus(self.low_factor),
                                                     high_freq_factor=F.softplus(self.high_factor),
                                                     split_ratio=self.split_ratio)
        dct_data_2 = self.frequency_noise_injection(dct_data,noise_factor=self.noise_factor, high_freq_emphasis=self.high_freq_emphasis)
        dct_data = dct_data_1 + dct_data_2
        idct_data = idct(dct_data, norm='ortho')
        idct_data = idct_data.to(x.dtype)

        return idct_data
