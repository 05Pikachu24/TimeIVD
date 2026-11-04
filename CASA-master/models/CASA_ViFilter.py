import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct, idct
import torch.nn.functional as F

class ScoreNetwork(nn.Module):
    def __init__(self, enc_in, d_model, kernel=3):
        super(ScoreNetwork, self).__init__()

        assert kernel % 2 == 1, "Kernel size must be an odd number"
        pad = kernel // 2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=enc_in, out_channels=int(d_model / 4), kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model / 4), out_channels=int(d_model / 2), kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model / 2), out_channels=d_model, kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=d_model, out_channels=int(d_model / 2), kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=int(d_model / 2), out_channels=int(d_model / 4), kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=int(d_model / 4), out_channels=enc_in, kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        skip = x
        decoded = self.decoder(encoded)
        output = decoded + skip[:, :, :decoded.size(2)]  # Adjusting the size if needed
        
        return output


class MLP_attention(nn.Module):
    def __init__(self, d_model, enc_in, kernel):
        super(MLP_attention, self).__init__()
        self.value = nn.Linear(d_model, d_model)
        self.score = ScoreNetwork(enc_in, d_model, kernel)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, L = x.shape
        x1 = torch.softmax(self.score(x), dim = 1) * self.value(x) + x
        x1 = self.norm(x1.reshape(-1, L)).reshape(B, N, L)
        # return self.mlp(x1) + x1
        return self.norm((self.mlp(x1) + x1).reshape(-1, L)).reshape(B, N, L)

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
        return: [batch_size, num_features, d_model]
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

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.model = nn.Sequential(
            # nn.Linear(self.seq_len, self.d_model),  # replace
            # todo ViFilter
            Patch_Diverse_V1(configs.seq_len, configs.d_model, configs.enc_in, configs.split_ratio, configs.noise_factor, configs.high_freq_emphasis),
            nn.ReLU(),
            *[
                nn.Sequential(
                    MLP_attention(configs.d_model, configs.enc_in, configs.kernel),
                    nn.ReLU()
                )
                for _ in range(configs.e_layers)
            ],
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # instance denorm
        y = y * torch.sqrt(seq_var) + seq_mean

        return y
