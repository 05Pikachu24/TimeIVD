import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct, idct

from utils.masking import generate_causal_mask, generate_self_only_mask, generate_partial_mask


class ChannelIndependence(nn.Module):
    def __init__(
            self,
    ):
        super(ChannelIndependence, self).__init__()

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        _, input_len, _ = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, input_len, 1)
        return x

class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super(AddSosTokenAndDropLast, self).__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  # [batch_size * num_features, 1, d_model]
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  # [batch_size * num_features, seq_len + 1, d_model]
        x = x[:, :-1, :]  # [batch_size * num_features, seq_len, d_model]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self, d_model: int, seq_len, num_heads: int, feedforward_dim: int, dropout: float, configs
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        # self.ff = nn.Sequential(
        #     nn.Linear(d_model, feedforward_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(feedforward_dim, d_model),
        # )

        self.IVD_MOE = TwoSIVD(
            num_variables=configs.enc_in,
            d_model=d_model,
            patch_num=seq_len,
            causal_dim=configs.Causal_dim, gaussian_dim=configs.Gaussian_dim, kernel_size=configs.kernel_size, context_dim=configs.context_dim, context_hidden_dim=configs.context_hidden_dim,
            num_experts=3
        )

        self.num_features = configs.enc_in

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))  # Epilepsy [96,25,512]

        # Feed-forward network  # todo original
        # ff_output = self.ff(x)
        # output = self.norm2(x + self.dropout(ff_output))

        output, moe_info = self.IVD_MOE(x.view(-1, self.num_features, x.shape[-2], x.shape[-1]))
        output = output.reshape(-1, output.shape[-2], output.shape[-1])

        return output, moe_info

class CausalTransformer(nn.Module):
    def __init__(
            self,
            d_model: int,
            seq_len,
            num_heads: int,
            num_layers: int,
            feedforward_dim: int,
            dropout: float,
            configs
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, seq_len, num_heads, feedforward_dim, dropout, configs)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        moe_infos = []
        for layer_idx, layer in enumerate(self.layers):
            x, moe_info = layer(x, mask)
            if isinstance(moe_info, dict):
                moe_info["layer"] = layer_idx + 1
            moe_infos.append(moe_info)

        x = self.norm(x)  # LN
        return x, moe_infos

class Patch_Diverse_V1(nn.Module):
    def __init__(self, dim, num_var, split_ratio):
        super().__init__()
        self.split_ratio = split_ratio
        self.num_var = num_var
        self.d_model = dim

        self.low_factor = nn.Parameter(torch.randn(num_var, int(self.d_model * split_ratio)))
        self.high_factor = nn.Parameter(torch.randn(num_var, self.d_model - int(self.d_model * split_ratio)))

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

        noise = torch.randn_like(freq_x)

        if high_freq_emphasis:
            seq_len = freq_x.shape[-1]
            emphasis = torch.linspace(0.1, 1.0, seq_len, device=freq_x.device).view(1, 1, 1, seq_len)
            noise = noise * emphasis
        return freq_x + noise_factor * noise

    def forward(self, x):

        BN, P, D = x.shape
        B = BN // self.num_var
        N = self.num_var

        x = x.reshape(-1, P, N, D).to(torch.float32)
        dct_data = dct(x, norm='ortho')

        dct_data_1 = self.manipulate_frequency_bands(dct_data, low_freq_factor=F.softplus(self.low_factor),
                                                     high_freq_factor=F.softplus(self.high_factor),
                                                     split_ratio=self.split_ratio)

        dct_data_2 = self.frequency_noise_injection(dct_data)

        dct_data = dct_data_1 + dct_data_2

        idct_data = idct(dct_data, norm='ortho')
        idct_data = idct_data.to(x.dtype)

        return idct_data

class TemporalVariableGating_V1(nn.Module):
    def __init__(self, num_variables, d_model, patch_num, kernel_size,num_experts):
        super().__init__()
        self.num_variables = num_variables
        self.num_experts = num_experts

        self.temporal_proj = nn.Sequential(
            nn.Conv1d(in_channels=num_variables, out_channels=num_variables, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.Linear(in_features=patch_num*d_model, out_features=patch_num)
        )

        self.gating_mlp = nn.Sequential(
            nn.Linear(num_variables, num_variables * 2),
            nn.ReLU(),
            nn.Linear(num_variables * 2, num_experts)
        )

    def forward(self, input_embedding):
        # B, P, N * D
        D = input_embedding.shape[-1] // self.num_variables
        input_embedding = input_embedding.view(-1, self.num_variables, input_embedding.shape[-2]*D)

        temporal_summary = self.temporal_proj(input_embedding)  # return [B,N,P]

        gating_logits = self.gating_mlp(temporal_summary.permute(0,2,1))  # [B, P, num_experts]

        return gating_logits

class HyperNetworkForMixing(nn.Module):
    def __init__(self, context_dim, num_vars, context_hidden_dim):
        super().__init__()
        self.num_vars = num_vars
        self.weight_generator = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Linear(context_hidden_dim, num_vars * num_vars)
        )
        self.bias_generator = nn.Linear(context_dim, num_vars)

    def forward(self, context_vec):
        weights = self.weight_generator(context_vec)
        biases = self.bias_generator(context_vec)

        return weights.view(-1, self.num_vars, self.num_vars), biases

class DynamicLinearVariateMixer(nn.Module):
    def __init__(self, num_variables, d_model, context_dim, context_hidden_dim):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model

        self.context_encoder = nn.Linear(d_model * num_variables, context_dim)

        self.hypernet = HyperNetworkForMixing(context_dim, num_variables, context_hidden_dim)
        self.norm = nn.LayerNorm(num_variables)

    def forward(self, x):
        B, P, N, D = x.shape

        context_input = x.reshape(B, P, D * N)
        context_vec = self.context_encoder(context_input.view(B * P, D * N))

        dynamic_W, dynamic_b = self.hypernet(context_vec)

        x_reshaped = x.reshape(B * P, D, N)

        output_reshaped = torch.bmm(x_reshaped, dynamic_W.transpose(1, 2)) + dynamic_b.unsqueeze(1)

        output_reshaped = self.norm(x.reshape(B * P, D, N) + output_reshaped)

        return output_reshaped.reshape(B, P, N, D)

class AffinityExpert(nn.Module):

    def __init__(self, num_variables, d_model, d_k=None):
        super().__init__()
        if d_k is None:
            d_k = d_model // 4

        self.d_k = d_k
        self.query_proj = nn.Linear(d_model, d_k)
        self.key_proj = nn.Linear(d_model, d_k)

    def forward(self, x):

        Q = self.query_proj(x)
        K = self.key_proj(x)

        affinity_matrix = torch.matmul(Q, K.transpose(-2, -1))

        affinity_matrix = affinity_matrix / (self.d_k ** 0.5)

        normalized_affinity = F.softmax(affinity_matrix, dim=-1)

        return normalized_affinity

class StaticDynamicAffinityExpert(nn.Module):

    def __init__(self, num_variables, d_model, d_k=None):
        super().__init__()
        self.num_variables = num_variables
        self.d_k = d_k if d_k is not None else d_model // 4

        self.static_structure = nn.Parameter(torch.randn(num_variables, num_variables))

        self.query_proj = nn.Linear(d_model, self.d_k)
        self.key_proj = nn.Linear(d_model, self.d_k)

    def forward(self, x):

        B, P, N, D = x.shape

        Q = self.query_proj(x)  # [B, P, N, d_k]
        K = self.key_proj(x)  # [B, P, N, d_k]
        dynamic_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # [N, N] -> [B, P, N, N]
        static_mask = torch.sigmoid(self.static_structure)
        static_mask = static_mask.unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1)

        masked_affinity = dynamic_weights * static_mask

        normalized_affinity = F.softmax(masked_affinity, dim=-1)

        return normalized_affinity

class GaussianDependencyDynamicPatched(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(GaussianDependencyDynamicPatched, self).__init__()
        self.embed_projector = nn.Linear(d_model, hidden_dim)

        self.scale = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        var_embeds = self.embed_projector(x)

        dist = torch.cdist(var_embeds, var_embeds, p=2)

        dep = torch.exp(-self.scale * dist ** 2)

        return dep


class TwoSIVD(nn.Module):
    def __init__(self, num_variables, d_model, patch_num, causal_dim, gaussian_dim, kernel_size, context_dim, context_hidden_dim, num_experts=4):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            AffinityExpert(num_variables, d_model),
            StaticDynamicAffinityExpert(num_variables, d_model),
            GaussianDependencyDynamicPatched(d_model, hidden_dim=gaussian_dim),
        ])

        self.gating_network = TemporalVariableGating_V1(num_variables, d_model, patch_num, kernel_size, num_experts)

        self.residual_proj = nn.Linear(d_model, d_model)

        self.aggregate_P = DynamicLinearVariateMixer(num_variables, d_model, context_dim, context_hidden_dim)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        B, N, P, D = x.shape

        # [B, N, P, D] -> [B, P, N, D]
        x_permuted = x.permute(0, 2, 1, 3)

        output_permuted = self.aggregate_P(x_permuted)

        gating_input = output_permuted.reshape(B, P, N * D)
        gating_logits = self.gating_network(gating_input)
        gating_weights = F.softmax(gating_logits, dim=-1)  # [B, P, num_experts]

        affinity_matrices = [expert(output_permuted) for expert in self.experts]

        # [num_experts, B, P, N, N]
        affinity_matrices_stacked = torch.stack(affinity_matrices, dim=0)

        gating_weights_reshaped = gating_weights.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)

        final_affinity = torch.sum(affinity_matrices_stacked * gating_weights_reshaped, dim=0)  # [B, P, N, N]

        # final_affinity: [B, P, N, N]
        # output_permuted:     [B, P, N, D]
        # aggregated_x:   [B, P, N, D]
        aggregated_x = torch.matmul(final_affinity, output_permuted)

        # output_permuted [B, P, N, D]
        output_permuted = x_permuted + self.residual_proj(aggregated_x)

        # [B, P, N, D] -> [B, N, P, D]
        final_output = output_permuted.permute(0, 2, 1, 3)

        moe_info = {
            "gating_weights": gating_weights,
            "expert_affinities": affinity_matrices_stacked.permute(1, 2, 0, 3, 4).contiguous(),
            "final_affinity": final_affinity,
        }

        return final_output, moe_info


class SparseDependency(nn.Module):
    def __init__(self, seq_len, hidden_dim, topk=6):
        super().__init__()
        self.query = nn.Linear(seq_len, hidden_dim)
        self.key = nn.Linear(seq_len, hidden_dim)
        self.topk = topk

    def forward(self, x):
        x = x.transpose(1, 2)
        Q, K = self.query(x), self.key(x)
        dep = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        values, indices = torch.topk(dep, self.topk, dim=-1)
        mask = torch.zeros_like(dep).scatter_(-1, indices, 1.0)
        dep = F.softmax(dep * mask, dim=-1)
        return dep

class Diffusion(nn.Module):
    def __init__(
            self,
            time_steps: int,
            device: torch.device,
            scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps  # 超参 默认 1000

        if scheduler == "cosine":
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler=}")

        self.alpha = 1 - self.betas
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
                torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps)
        return betas

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(x.shape[:2])  # [batch_size * num_features, seq_len]
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tgt_mask, src_mask):

        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query = self.norm1(query + self.dropout(attn_output))

        # Encoder attention
        attn_output, _ = self.encoder_attention(query, key, value, attn_mask=src_mask)
        query = self.norm2(query + self.dropout(attn_output))

        x = query

        return x


class DenoisingPatchDecoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_layers: int,
            feedforward_dim: int,
            dropout: float,
            mask_ratio: float,
    ):
        super(DenoisingPatchDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.mask_ratio = mask_ratio

    def forward(self, query, key, value, is_tgt_mask=True, is_src_mask=True):
        seq_len = query.size(1)
        tgt_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_tgt_mask else None
        )
        src_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_src_mask else None
        )
        for layer in self.layers:
            query = layer(query, key, value, tgt_mask, src_mask)
        x = self.norm(query)
        return x


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(self.receptive_field - 1),  # 左填充
            dilation=dilation,
            groups=groups
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class CausalTCN(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, kernel_size=3):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth

        # First linear layer to map input_dims to hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        # Create a dilated causal convolutional encoder
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * (depth - 1) + [output_dims],
            kernel_size=kernel_size
        )

    def forward(self, x):
        # Input x is of shape [batch_size, seq_len, input_dims]

        # Flatten input (batch_size, seq_len, input_dims) -> (batch_size, seq_len, hidden_dims)
        x = self.input_fc(x)

        # Transpose for the convolution (batch_size, seq_len, hidden_dims) -> (batch_size, hidden_dims, seq_len)
        x = x.transpose(1, 2)

        # Apply dilated convolutions
        x = self.feature_extractor(x)  # [batch_size, hidden_dims, seq_len] -> [batch_size, output_dims, seq_len]

        # Transpose back to [batch_size, seq_len, output_dims]
        x = x.transpose(1, 2)

        return x


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, input_dims]
        :return: [batch_size, seq_len, output_dims]
        """
        x = x.transpose(1, 2)
        return self.net(x).transpose(1, 2)


class ClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(ClsHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * d_model, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


class OldClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(OldClsHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(torch.max(x, dim=1)[0])


class ClsEmbedding(nn.Module):
    def __init__(self, num_features, d_model, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x): # [32,206,3]
        x = x.transpose(1, 2)
        return self.conv(x).transpose(1, 2)


class ClsFlattenHead(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, num_features, dropout):
        super(ClsFlattenHead, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len * num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # [batch_size, seq_len * d_model]
        x = self.dropout(x)  # [batch_size, seq_len * d_model]
        x = self.forecast_head(x)  # [batch_size, pred_len * num_features]
        return x.reshape(x.size(0), self.pred_len, self.num_features)


class ARFlattenHead(nn.Module):
    def __init__(
            self,
            d_model: int,
            patch_len: int,
            dropout: float,
    ):
        super(ARFlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, seq_len * patch_len, num_features]
        """
        x = self.forecast_head(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.dropout(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.flatten(x)  # (batch_size, num_features, seq_len * patch_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len * patch_len, num_features)
        return x
