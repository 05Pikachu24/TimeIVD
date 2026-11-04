import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import math
import torch.fft
from einops import rearrange

# TwoSIVD -----------------------------------------------------
import torch
class TwoSIVD(nn.Module):
    def __init__(self, num_variables, d_model, gaussian_dim, kernel_size, context_dim, context_hidden_dim, num_experts=4, use_hyper=False, use_moe=True):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model
        self.num_experts = num_experts
        self.use_hyper = use_hyper
        self.use_moe = use_moe
        self.experts = nn.ModuleList([
            AffinityExpert(num_variables, d_model),
            StaticDynamicAffinityExpert(num_variables, d_model),
            GaussianDependencyDynamicPatched(d_model, hidden_dim=gaussian_dim),
        ])

        self.gating_network = TemporalVariableGating_V1(num_variables, d_model, kernel_size, num_experts)

        self.residual_proj = nn.Linear(d_model, d_model)
        self.aggregate_P = DynamicLinearVariateMixer(num_variables, d_model, context_dim, context_hidden_dim)
        # self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        """
        input: [B, N, D]
        output: [B, N, D]
        """
        B, N, D = x.shape

        x_permuted = x  # .permute(0, 2, 1, 3)

        if self.use_hyper:
            output_permuted = self.aggregate_P(x_permuted)   # [B,N,D]  remove HyperNet
        else:
            output_permuted = x_permuted
        if self.use_moe:
            gating_input = output_permuted.reshape(B, N * D)
            gating_logits = self.gating_network(gating_input)
            gating_weights = F.softmax(gating_logits, dim=-1)  # [B, 1, num_experts]

            affinity_matrices = [expert(output_permuted) for expert in self.experts]

            # [num_experts, B, P, N, N]
            affinity_matrices_stacked = torch.stack(affinity_matrices, dim=0)  # [3,B,N,N]

            # gating_weights [B, 1, num_experts] -> [num_experts, B, 1, 1]
            gating_weights_reshaped = gating_weights.squeeze(1).permute(1,0).unsqueeze(-1).unsqueeze(-1)

            final_affinity = torch.sum(affinity_matrices_stacked * gating_weights_reshaped, dim=0)  # [B, N, N]

            aggregated_x = torch.matmul(final_affinity, output_permuted)

            # output_permuted [B, N, D]
            output_permuted = self.residual_proj(aggregated_x)
        # output_permuted = aggregated_x
        else:
            output_permuted = output_permuted
        # [B, N, D] -> [B, N, D]
        final_output = output_permuted  # .permute(0, 2, 1, 3)

        return final_output
class GaussianDependencyDynamicPatched(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(GaussianDependencyDynamicPatched, self).__init__()
        self.embed_projector = nn.Linear(d_model, hidden_dim)

        self.scale = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        # x [B, N, L] -> [B, N, hidden_dim]
        var_embeds = self.embed_projector(x)

        # dist [B, N, N]
        dist = torch.cdist(var_embeds, var_embeds, p=2)

        # 应用高斯核函数得到依赖矩阵 [B, N, N]
        dep = torch.exp(-self.scale * dist ** 2)

        return dep
class TemporalVariableGating_V1(nn.Module):
    def __init__(self, num_variables, d_model, kernel_size,num_experts):
        super().__init__()
        self.num_variables = num_variables
        self.num_experts = num_experts

        self.temporal_proj = nn.Sequential(
            nn.Conv1d(in_channels=num_variables, out_channels=num_variables, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
                nn.Linear(in_features=d_model, out_features=1)
        )

        self.gating_mlp = nn.Sequential(
            nn.Linear(num_variables, num_variables * 2),
            nn.ReLU(),
            nn.Linear(num_variables * 2, num_experts)
        )

    def forward(self, input_embedding):
        # B, N * D
        D = input_embedding.shape[-1] // self.num_variables
        input_embedding = input_embedding.view(-1, self.num_variables, D)

        temporal_summary = self.temporal_proj(input_embedding)  # return [B,N,1]

        gating_logits = self.gating_mlp(temporal_summary.permute(0,2,1))  # [B, 1, num_experts]

        return gating_logits
class AffinityExpert(nn.Module):

    def __init__(self, num_variables, d_model, d_k=None):
        super().__init__()
        if d_k is None:
            d_k = d_model // 4

        self.d_k = d_k
        self.query_proj = nn.Linear(d_model, d_k)
        self.key_proj = nn.Linear(d_model, d_k)

    def forward(self, x):
        """
        input: [B, N, D]
        output: [B, N, N]
        # x: [Batch, Num_variables, D_model]
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)

        # Q @ K^T
        # Q:           [B, N, d_k]
        # K.transpose: [B, d_k, N]
        # output:      [B, N, N]
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
        """
        input: [B, N, D]
        output: [B, N, N]
        """
        B, N, D = x.shape

        Q = self.query_proj(x)  # [B, N, d_k]
        K = self.key_proj(x)  # [B, N, d_k]
        dynamic_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # [N, N] -> [B, N, N]
        static_mask = torch.sigmoid(self.static_structure)
        static_mask = static_mask.unsqueeze(0).expand(B, -1, -1)

        masked_affinity = dynamic_weights * static_mask

        normalized_affinity = F.softmax(masked_affinity, dim=-1)

        return normalized_affinity


class DynamicLinearVariateMixer(nn.Module):
    def __init__(self, num_variables, d_model, context_dim, context_hidden_dim):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model

        self.context_encoder = nn.Linear(d_model * num_variables, context_dim)  # todo 128

        self.hypernet = HyperNetworkForMixing(context_dim, num_variables, context_hidden_dim)
        self.norm = nn.LayerNorm(num_variables)

    def forward(self, x):
        B, N, D = x.shape

        context_input = x.reshape(B, D * N)
        context_vec = self.context_encoder(context_input.view(B, D * N))

        dynamic_W, dynamic_b = self.hypernet(context_vec)
        x_reshaped = x.reshape(B, D, N)
        output_reshaped = torch.bmm(x_reshaped, dynamic_W.transpose(1, 2)) + dynamic_b.unsqueeze(1)

        output_reshaped = self.norm(x.reshape(B, D, N) + output_reshaped)

        return output_reshaped.reshape(B, N, D)


class HyperNetworkForMixing(nn.Module):
    def __init__(self, context_dim, num_vars, context_hidden_dim):
        super().__init__()
        self.num_vars = num_vars
        self.weight_generator = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),  # todo 128
            nn.ReLU(),
            nn.Linear(context_hidden_dim, num_vars * num_vars)
        )
        self.bias_generator = nn.Linear(context_dim, num_vars)

    def forward(self, context_vec):
        # W_p[B*P, N*N]
        weights = self.weight_generator(context_vec)
        # b_p [B*P, N]
        biases = self.bias_generator(context_vec)

        # [B*P, N, N]
        return weights.view(-1, self.num_vars, self.num_vars), biases
# TwoSIVD -----------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class EncoderLayer_TwoSIVD(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, config=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_TwoSIVD, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.IVD_MOE = TwoSIVD(
            num_variables=config.enc_in,
            d_model=d_model, gaussian_dim=config.Gaussian_dim, kernel_size=config.kernel_size, context_dim=config.context_dim, context_hidden_dim=config.context_hidden_dim,
            num_experts=3,   #
            use_hyper= config.use_hyper,
            use_moe= config.use_moe,
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        output = self.IVD_MOE(y)
        y = self.dropout(output)
        # y = y.permute(0, 2, 1)

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class Encoder_TwoSIVD(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder_TwoSIVD, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn, adj = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = True  # output_attention  # reset to True  todo verifying
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # if self.mask_flag:
        #     large_negative = -math.log(1e10)
        #     attention_mask = torch.where(attn_mask == 0, torch.tensor(large_negative), attn_mask)
        #
        #     scores = scores * attention_mask
        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)

            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class FullAttention_DGL_CAL(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_DGL_CAL, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = True # output_attention  # reset to True  todo verifying
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # if self.mask_flag:
        #     large_negative = -math.log(1e10)
        #     attention_mask = torch.where(attn_mask == 0, torch.tensor(large_negative), attn_mask)
        #
        #     scores = scores * attention_mask
        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)

            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size):
        super(Mahalanobis_mask, self).__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(torch.randn(frequency_size, frequency_size), requires_grad=True)

    def calculate_prob_distance(self, X):
        XF = torch.abs(torch.fft.rfft(X, dim=-1))
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)

        # B x C x C x D
        diff = X1 - X2

        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)

        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)

        # exp_dist = torch.exp(-dist)
        exp_dist = 1 / (dist + 1e-10)
        # 对角线置零

        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()

        # B x C x C
        p = exp_dist / exp_max

        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)

        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99

        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)

        # bernoulli中两个通道有关系的概率
        sample = self.bernoulli_gumbel_rsample(p)

        mask = sample.unsqueeze(1)
        cnt = torch.sum(mask, dim=-1)
        return mask
