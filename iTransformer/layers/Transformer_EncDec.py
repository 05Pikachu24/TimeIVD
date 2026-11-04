import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


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

class EncoderLayer_IVD(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, configs=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_IVD, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.IVD_MOE = TwoSIVD(
            num_variables=configs.enc_in,
            d_model=d_model, gaussian_dim=configs.Gaussian_dim, kernel_size=configs.kernel_size, context_dim=configs.context_dim, context_hidden_dim=configs.context_hidden_dim,
            num_experts=3   #
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

        # TwoSIVD  [B,N,L]
        y = self.IVD_MOE(y)
        return self.norm2(x + y), attn  # originl

# TwoSIVD -----------------------------------------------------
import torch
class TwoSIVD(nn.Module):
    def __init__(self, num_variables, d_model, gaussian_dim, kernel_size, context_dim, context_hidden_dim, num_experts=4):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            AffinityExpert(num_variables, d_model),
            StaticDynamicAffinityExpert(num_variables, d_model),
            GaussianDependencyDynamicPatched(d_model, hidden_dim=gaussian_dim),
        ])

        self.gating_network = TemporalVariableGating_V1(num_variables, d_model, kernel_size, num_experts)
        self.residual_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        input: [B, N, D]
        output: [B, N, D]
        """
        B, N, D = x.shape

        x_permuted = x

        # output_permuted = self.aggregate_P(x_permuted)   # [B,N,D]  remove HyperNet
        output_permuted = x_permuted
        gating_input = output_permuted.reshape(B, N * D)
        gating_logits = self.gating_network(gating_input)  # todo
        gating_weights = F.softmax(gating_logits, dim=-1)  # [B, 1, num_experts]

        affinity_matrices = [expert(output_permuted) for expert in self.experts]

        # [num_experts, B, P, N, N]
        affinity_matrices_stacked = torch.stack(affinity_matrices, dim=0)  # [3,B,N,N]

        # gating_weights [B, 1, num_experts] -> [num_experts, B, 1, 1]
        gating_weights_reshaped = gating_weights.squeeze(1).permute(1,0).unsqueeze(-1).unsqueeze(-1)

        final_affinity = torch.sum(affinity_matrices_stacked * gating_weights_reshaped, dim=0)  # [B, N, N]

        aggregated_x = torch.matmul(final_affinity, output_permuted)

        # output_permuted [B, N, D]
        output_permuted = self.residual_proj(aggregated_x)  # todo 有效
        # output_permuted = aggregated_x

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

        # [B, N, N]
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
        """ 输入 x: [B, P, N, D] -> 输出: [B, P, N, D] """
        B, N, D = x.shape

        context_input = x.reshape(B, D * N)
        context_vec = self.context_encoder(context_input.view(B, D * N))

        dynamic_W, dynamic_b = self.hypernet(context_vec)
        x_reshaped = x.reshape(B, D, N)
        output_reshaped = torch.bmm(x_reshaped, dynamic_W.transpose(1, 2)) + dynamic_b.unsqueeze(1)

        output_reshaped = self.norm(x.reshape(B, D, N) + output_reshaped)  # todo verifying

        return output_reshaped.reshape(B, N, D)

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
# TwoSIVD -----------------------------------------------------

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


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
