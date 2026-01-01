import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_uncertainty(prob_map, eps=1e-6):
    p = prob_map.clamp(min=eps, max=1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / math.log(2.0)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, query, context):
        batch_size, _, dim = query.shape
        q = self.to_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        prob = attn.softmax(dim=-1)
        out = torch.matmul(prob, v).transpose(1, 2).reshape(batch_size, -1, dim)
        return self.to_out(out)


class UGHRHyperedgeGenerator(nn.Module):
    def __init__(
        self,
        node_dim,
        num_fg_prototypes,
        num_bg_prototypes,
        num_heads=4,
        dropout=0.1,
        logit_scale=1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_fg_prototypes = num_fg_prototypes
        self.num_bg_prototypes = num_bg_prototypes
        self.total_prototypes = num_fg_prototypes + num_bg_prototypes

        self.head_dim = node_dim // num_heads
        self.scaling = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

        self.prototype_base_fg = nn.Parameter(torch.Tensor(num_fg_prototypes, node_dim))
        self.prototype_base_bg = nn.Parameter(torch.Tensor(num_bg_prototypes, node_dim))
        nn.init.xavier_uniform_(self.prototype_base_fg)
        nn.init.xavier_uniform_(self.prototype_base_bg)

        context_dim = 2 * node_dim
        self.attn_fg_to_bg = CrossAttention(context_dim, num_heads=num_heads)
        self.attn_bg_to_fg = CrossAttention(context_dim, num_heads=num_heads)

        self.context_net_fg = nn.Linear(context_dim, num_fg_prototypes * node_dim)
        self.context_net_bg = nn.Linear(context_dim, num_bg_prototypes * node_dim)

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.logit_scale = logit_scale

    def forward(self, node_tokens, coarse_prob):
        batch_size, num_nodes, dim = node_tokens.shape
        height = width = int(num_nodes ** 0.5)

        node_2d = node_tokens.transpose(1, 2).view(batch_size, dim, height, width)
        fg_mask = coarse_prob
        bg_mask = 1.0 - fg_mask

        fg_count = fg_mask.sum(dim=[2, 3]) + 1e-6
        bg_count = bg_mask.sum(dim=[2, 3]) + 1e-6

        fg_avg = (node_2d * fg_mask).sum(dim=[2, 3]) / fg_count
        bg_avg = (node_2d * bg_mask).sum(dim=[2, 3]) / bg_count

        fg_bin = fg_mask > 0.5
        fill_val = torch.finfo(node_2d.dtype).min
        fg_max_vals = node_2d.masked_fill(~fg_bin.expand_as(node_2d), fill_val).flatten(2).amax(dim=-1)
        has_fg = fg_bin.flatten(2).any(dim=-1)
        global_max = node_2d.flatten(2).amax(dim=-1)
        fg_max = torch.where(has_fg, fg_max_vals, global_max)

        bg_bin = ~fg_bin
        bg_max_vals = node_2d.masked_fill(~bg_bin.expand_as(node_2d), fill_val).flatten(2).amax(dim=-1)
        has_bg = bg_bin.flatten(2).any(dim=-1)
        bg_max = torch.where(has_bg, bg_max_vals, global_max)

        fg_context = torch.cat([fg_avg, fg_max], dim=-1).unsqueeze(1)
        bg_context = torch.cat([bg_avg, bg_max], dim=-1).unsqueeze(1)

        ctx_fg = self.attn_fg_to_bg(fg_context, bg_context)
        ctx_bg = self.attn_bg_to_fg(bg_context, fg_context)

        fg_offset = self.context_net_fg(ctx_fg.squeeze(1)).view(batch_size, self.num_fg_prototypes, dim)
        bg_offset = self.context_net_bg(ctx_bg.squeeze(1)).view(batch_size, self.num_bg_prototypes, dim)

        prototypes = torch.cat(
            [
                self.prototype_base_fg.unsqueeze(0) + fg_offset,
                self.prototype_base_bg.unsqueeze(0) + bg_offset,
            ],
            dim=1,
        )

        node_proj = self.pre_head_proj(node_tokens)
        node_heads = node_proj.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(batch_size, self.total_prototypes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        node_flat = node_heads.reshape(batch_size * self.num_heads, num_nodes, self.head_dim)
        proto_flat = proto_heads.reshape(batch_size * self.num_heads, self.total_prototypes, self.head_dim).transpose(1, 2)
        logits = torch.bmm(node_flat, proto_flat) / self.scaling
        logits = logits.view(batch_size, self.num_heads, num_nodes, self.total_prototypes).mean(dim=1)
        logits = self.dropout(logits)

        uncertainty = entropy_uncertainty(coarse_prob).flatten(1).unsqueeze(-1)
        logits = logits * (1.0 + self.logit_scale * uncertainty)

        return F.softmax(logits, dim=1)


class UGHRBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_fg_prototypes=8,
        num_bg_prototypes=8,
        num_heads=4,
        dropout=0.1,
        logit_scale=1.0,
    ):
        super().__init__()
        self.edge_generator = UGHRHyperedgeGenerator(
            embed_dim,
            num_fg_prototypes,
            num_bg_prototypes,
            num_heads,
            dropout,
            logit_scale,
        )
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, node_tokens, coarse_prob):
        participation = self.edge_generator(node_tokens, coarse_prob)
        hyperedges = torch.bmm(participation.transpose(1, 2), node_tokens)
        hyperedges = self.edge_proj(hyperedges)
        node_update = torch.bmm(participation, hyperedges)
        node_update = self.node_proj(node_update)
        return node_tokens + node_update
