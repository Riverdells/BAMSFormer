import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F
from lib.MyGraph import MyGraph
import math
import torch.nn.init as init

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
def compute_time_step_correlation(x):
    B, T, N, D = x.shape
    correlation_matrix = torch.zeros(B, T, T).to(x.device)
    x_flat = x.view(B, T, -1)
    for t in range(0, T):
        current_time_step_flat = x_flat[:, t, :]
        correlation = F.cosine_similarity(current_time_step_flat.unsqueeze(1), x_flat[:, :, :], dim=-1)
        correlation_matrix[:, t, :] = correlation
    correlation_matrix = torch.triu(correlation_matrix)
    return correlation_matrix

class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self,
                 num_nodes,
                 time_steps,
                 embed_dim):
        super(SpatioTemporalPositionalEncoding, self).__init__()
        self.spatial_embedding = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.empty(time_steps, embed_dim))
        self.fusion_weight = nn.Parameter(torch.empty(1, embed_dim))
        self._reset_parameters()
        self.time_steps = time_steps

    def _reset_parameters(self):
        init.xavier_uniform_(self.spatial_embedding)
        init.xavier_uniform_(self.temporal_embedding)
        init.xavier_uniform_(self.fusion_weight)
    def forward(self, batch_size):
        spatial = self.spatial_embedding.unsqueeze(0).unsqueeze(0)
        st_encoding = spatial
        st_encoding = st_encoding.expand(batch_size, self.time_steps, -1,-1)
        return st_encoding

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()

class BasicSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads0=2, num_heads1=4, num_heads2=4, mask=False,
                 ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads0 = num_heads0
        self.num_heads1 = num_heads1
        self.num_heads2 = num_heads2
        self.mask = mask
        self.ra0 = self.num_heads0 / (self.num_heads0 + self.num_heads1 + self.num_heads2)
        self.ra1 = self.num_heads1 / (self.num_heads0 + self.num_heads1 + self.num_heads2)
        self.ra2 = self.num_heads2 / (self.num_heads0 + self.num_heads1 + self.num_heads2)
        self.head_dim = model_dim // (self.num_heads0 + self.num_heads1 + self.num_heads2)
        self.scale = self.head_dim ** -0.5

        self.FC_Q = nn.Linear(model_dim, int(self.model_dim * self.ra0))
        self.FC_K = nn.Linear(model_dim, int(self.model_dim * self.ra0))
        self.FC_V = nn.Linear(model_dim, int(self.model_dim * self.ra0))

        self.t_q_conv = nn.Conv2d(model_dim, int(self.model_dim * self.ra1), kernel_size=1, bias=False)
        self.t_k_conv = nn.Conv2d(model_dim, int(self.model_dim * self.ra1), kernel_size=1, bias=False)
        self.t_v_conv = nn.Conv2d(model_dim, int(self.model_dim * self.ra1), kernel_size=1, bias=False)

        self.geo_q_conv = nn.Conv2d(model_dim, int(self.model_dim * self.ra2), kernel_size=1, bias=False)
        self.geo_k_conv = nn.Conv2d(model_dim, int(self.model_dim * self.ra2), kernel_size=1, bias=False)
        self.geo_v_conv = nn.Conv2d(model_dim, int(self.model_dim * self.ra2), kernel_size=1, bias=False)
        self.geo_attn_drop = nn.Dropout(0.1)
        self.t_attn_drop = nn.Dropout(0.1)
        self.out_proj = nn.Linear(int(model_dim), int(model_dim))
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, query, key, value, pos, geo_mask=None, flag=1):
        if flag == 1:
            x = query.permute(0, 2, 1, 3)
        else:
            x = query
        B, T, N, C = x.shape
        batch_size = query.shape[0]
        p = pos[0, :, :, :, :]
        if self.num_heads0 != 0:
            query = self.FC_Q(query)
            key = self.FC_K(key)
            value = self.FC_V(value)
            query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
            key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
            value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
            key = key.transpose(-1, -2)
            attn_score = (query @ key) / self.head_dim ** 0.5
            attn_score = torch.softmax(attn_score, dim=-1)
            out = attn_score @ value
            out1 = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)

        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.num_heads1, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.num_heads1, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.num_heads1, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn + pos.transpose(-2,-1)
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(C * self.ra1))

        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, N, self.num_heads2, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.num_heads2, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.num_heads2, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale

        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, N, T, int(C * self.ra2))
        if flag == 2:
            t_x = t_x.permute(0, 2, 1, 3)
            geo_x = geo_x.permute(0, 2, 1, 3)
        out = self.out_proj(torch.cat([out1, t_x, geo_x], dim=-1))
        out = self.proj_drop(out)
        return out


class BasicLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads0=8, num_heads1=8, num_heads2=8, dropout=0,
            mask=False
    ):
        super().__init__()
        self.num_heads0 = num_heads0
        self.num_heads1 = num_heads1
        self.num_heads2 = num_heads2

        self.attn = BasicSelfAttention(model_dim, num_heads0, num_heads1, num_heads2, mask)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.D1 = DyT(model_dim)
        self.D2 = DyT(model_dim)

    def forward(self, x, pos, dim=-2, geo_mask=None):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x, pos, geo_mask, flag=dim)
        out = self.dropout1(out)
        out = self.D1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.D2(residual + out)
        out = out.transpose(dim, -2)
        return out

class BAMSFormer(nn.Module):
    def __init__(
            self,
            dataset,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            mod_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            hhod_embedding_dim=24,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads0=4,
            num_heads1=4,
            num_heads2=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
            embed_dim=24,
            supports=None,
            dropout1=0.01,
            dim=1,
    ):
        super().__init__()
        self.graph = MyGraph(dataset)

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.mod_embedding_dim = mod_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.hhod_embedding_dim = hhod_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.embed_dim = embed_dim
        self.dim = dim
        self.feed_forward_dim = feed_forward_dim
        self.dropout = dropout
        self.dropout1 = dropout1
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + embed_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
                +12
        )
        self.num_heads0 = num_heads0
        self.num_heads1 = num_heads1
        self.num_heads2 = num_heads2
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.input_proj = nn.Linear(input_dim, 24)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.hhod_embedding = nn.Embedding(48, dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.position_encoding_APE = PositionalEncoding(self.input_embedding_dim)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                BasicLayer(self.model_dim, self.feed_forward_dim, self.num_heads0, self.num_heads1, self.num_heads2,
                           self.dropout)
                for _ in range(num_layers)
            ]
        )

        self.proj = nn.Linear(self.model_dim, self.model_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.geo_masks, self.adj = self.graph.compute_masks()
        self.geo_mask = None
        self.dropout = nn.Dropout(dropout1)
        self.holiday_embedding = nn.Embedding(2, dow_embedding_dim)

    def forward(self, x):
        source = x
        batch_size = x.shape[0]
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]

        x = x[..., : self.input_dim]
        x = self.input_proj(x)
        x1 = x
        features = []
        if self.input_embedding_dim > 0:
            features.append(x)
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * 288).long()
            )
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            hhod_emb = self.hhod_embedding(
                (tod * 48).long()
            )
            features.append(hhod_emb)
        hod_tiled = source[..., 3]
        holiday_encoded = self.holiday_embedding(hod_tiled.long())
        features.append(holiday_encoded)
        if self.spatial_embedding_dim > 0:
            features.append(self.position_encoding_APE(x))
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(self.dropout(adp_emb))
        x = torch.cat(features, dim=-1)
        pos = compute_time_step_correlation(x1)
        pos = pos.unsqueeze(1)
        pos = pos.unsqueeze(2)
        for i, attn in enumerate(self.attn_layers_t):
            self.geo_mask = self.geo_masks[i]
            x = attn(x,pos=pos, dim=2, geo_mask=self.geo_mask)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)
        else:
            out = x.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))
        return out

if __name__ == "__main__":
    model = BAMSFormer(207, 12, 12)
    summary(model, [64, 12, 207, 3])


