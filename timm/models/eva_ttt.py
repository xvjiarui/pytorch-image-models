""" EVA

EVA from https://github.com/baaivision/EVA , paper: https://arxiv.org/abs/2211.07636

@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang,
  Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}

EVA-02: A Visual Representation for Neon Genesis - https://arxiv.org/abs/2303.11331
@article{EVA02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}

This file contains EVA & EVA02 model implementations evolved from BEiT, additional models in vision_transformer.py.

Modifications by / Copyright 2023 Ross Wightman, original copyrights below
"""
# EVA models Copyright (c) 2022 BAAI-Vision
# EVA02 models Copyright (c) 2023 BAAI-Vision
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.optim as optim

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn

from torch.utils._pytree import tree_map
import addict

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._registry import generate_default_cfgs, register_model

__all__ = ['EvaTTT']


class LayerNormFp32(LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = super().forward(x.to(torch.float32))
        return x.to(orig_type)


def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."
    orig_type = x.dtype
    x = x.to(torch.float32)

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y.to(orig_type)


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    orig_type = x.dtype
    x = x.to(torch.float32)

    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z.to(orig_type)


# Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out

class OnlineSGD:

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        
    def step_online(self, gradients):

        updated_params = []
        for i, param in enumerate(self.params):
            d_p = gradients[i]
            updated_param = param.add(d_p, alpha=-self.lr)
            updated_params.append(updated_param)

        
        return updated_params


class TTTWrapper(nn.Module):

    def __init__(
        self,
        dim: int,
        inner_model: nn.Module,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        ttt_base_lr: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.num_prefix_tokens = num_prefix_tokens

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.ttt_base_lr = ttt_base_lr
        # [width, 1]
        linear_weight_data = nn.Linear(dim, 1, bias=True).weight.data
        # prepending head dim -> [num_heads, width, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack([torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(num_heads)], dim=0)
        )
        linear_bias_data = nn.Linear(dim, 1, bias=True).bias.data
        # init bias to 0 following original JAX impl.
        # [num_heads, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack([torch.zeros_like(linear_bias_data) for _ in range(num_heads)], dim=0)
        )

        ln_weight_data = nn.LayerNorm(head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (num_heads, 1)))
        ln_bias_data = nn.LayerNorm(head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (num_heads, 1)))
    
        
        self.inner_model = inner_model

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)
        
        ttt_norm_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, -1)
        ttt_norm_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, -1)

        reconstruction_target = v - k
        # [B, num_heads, N, 1]
        ttt_lr_eta = self.get_eta(x) / q.shape[-1]

        with torch.enable_grad():
            inner_meta = InnerMeta(self.inner_model, B)
            self_params = inner_meta.to_self_parms()

            train_pred = self.inner_model(k, self_params)
            train_pred = ln_fwd(train_pred, ttt_norm_weight, ttt_norm_bias)
            mse_loss = 0.5 * (train_pred - reconstruction_target) ** 2
            mse_loss = mse_loss * ttt_lr_eta
            mse_loss = mse_loss.sum() / N
            gradients = torch.autograd.grad(
                mse_loss, inner_meta.params, create_graph=True
            )
            optim = OnlineSGD(inner_meta.params, lr=self.ttt_base_lr)
            updated_inner_params = optim.step_online(gradients)
        for (name, _), param in zip(self.inner_model.named_parameters(), updated_inner_params):
            self_params[name] = param

        test_pred = self.inner_model(q, self_params)
        test_pred = ln_fwd(test_pred, ttt_norm_weight, ttt_norm_bias)
        x = test_pred + q

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_eta(self, x):
        B, N, C = x.shape

        # [B, num_heads, N, 1]
        ttt_lr = torch.einsum("bnc,hdc->bhnd", x, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        ttt_lr = self.ttt_base_lr * ttt_lr

        return ttt_lr

class InnerMeta:
    def __init__(self, model, batch_size):
        self.model = model
        self.named_params = dict()
        self.params = []
        for name, param in model.named_parameters():
            batch_param = torch.repeat_interleave(param.unsqueeze(0), batch_size, dim=0)
            self.named_params[name] = batch_param
            self.params.append(batch_param)
    
    def to_self_parms(self):
        return addict.Dict(self.named_params)


class InnerLinear(nn.Module):
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        # inner loop weight matrix initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        # inner loop bias initialization
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def forward(self, x, self_params=None):
        """
        Args:
            x (torch.Tensor): [B, num_heads, N, head_dim]
        """
        # [B, nh, K, f] @ [B, nh, f, f]
        Z1 = (x @ self_params.W1) + self_params.b1

        return Z1


class InnerAttention(nn.Module):
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        # inner loop weight matrix initialization for TTT-Linear
        self.wqkv = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim * 3)))
        # inner loop bias initialization
        self.bqkv = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim *3))


    def forward(self, x, self_params=None):
        """
        Args:
            x (torch.Tensor): [B, num_heads, N, head_dim]
        """
        B, num_heads, N, head_dim = x.shape

        # [B, nh, K, f] @ [B, nh, f, 3f]
        qkv = (x @ self_params.wqkv) + self_params.bqkv
        qkv = qkv.reshape(B, num_heads, N, head_dim, 3)
        q, k, v = qkv.unbind(-1)
        # x = F.scaled_dot_product_attention(q, k, v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        return x
        

class EvaBasePrimal(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
            ttt_base_lr: float = 1.0,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.num_prefix_tokens = num_prefix_tokens

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # [width, 1]
        linear_weight_data = nn.Linear(dim, 1, bias=True).weight.data
        # prepending head dim -> [num_heads, width, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack([torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(num_heads)], dim=0)
        )
        linear_bias_data = nn.Linear(dim, 1, bias=True).bias.data
        # init bias to 0 following original JAX impl.
        # [num_heads, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack([torch.zeros_like(linear_bias_data) for _ in range(num_heads)], dim=0)
        )

        ln_weight_data = nn.LayerNorm(head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (num_heads, 1)))
        ln_bias_data = nn.LayerNorm(head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (num_heads, 1)))

        self.num_mini_batch = 1
        self.ttt_base_lr = ttt_base_lr
    
    def extra_repr(self) -> str:
        return f"num_mini_batch={self.num_mini_batch}, ttt_base_lr={self.ttt_base_lr}"

    def ttt(self, q, k, v, ttt_lr_eta):
        raise NotImplementedError

    def get_eta(self, x):
        B, N, C = x.shape

        # [B, num_heads, N, 1]
        ttt_lr = torch.einsum("bnc,hdc->bhnd", x, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        ttt_lr = self.ttt_base_lr * ttt_lr

        return ttt_lr

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        # [B, num_heads, N, 1]
        ttt_lr_eta = self.get_eta(x) / q.shape[-1]
        # [B, num_heads, N, head_dim]
        x = self.ttt(q, k, v, ttt_lr_eta)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaM1Primal(EvaBasePrimal):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # inner loop weight matrix initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        # inner loop bias initialization
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
    

    def ttt(self, q, k, v, ttt_lr_eta):
        """
        Args:
            q: [B, num_heads, N, head_dim]
            k: [B, num_heads, N, head_dim]
            v: [B, num_heads, N, head_dim]
            ttt_lr_eta: [B, num_heads, N, 1]
        """

        B, num_heads, N, head_dim = q.shape
        num_mini_batch = self.num_mini_batch
        mini_batch_size = N // num_mini_batch
        device = q.device
        dtype = q.dtype


        def compute_mini_batch(params_dic, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim
            W1_init = params_dic["W1_states"]
            # [B, nh, 1, f]
            b1_init = params_dic["b1_states"]

            # [B,nh,K,f], K=mini_batch_size
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]
            mini_batch_size = ttt_lr_eta_mini_batch.shape[-2]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(num_heads, 1, head_dim)
            ln_bias = self.ttt_norm_bias.reshape(num_heads, 1, head_dim)
            # [B,nh,K,f]
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)
            grad_l_wrt_Z1 = grad_l_wrt_Z1 * ttt_lr_eta_mini_batch / mini_batch_size

            # [B, nh, f, f]
            grad_W1 = X1.transpose(-2, -1) @ grad_l_wrt_Z1
            # [B, nh, 1, f]
            grad_b1 = torch.sum(grad_l_wrt_Z1, dim=2, keepdim=True)

            W1_bar = W1_init - grad_W1
            b1_bar = b1_init - grad_b1

            # [B, nh, K, f] @ [B, nh, f, f]
            Z1_bar = (XQ_mini_batch @ W1_bar) + b1_bar

            W1_last = W1_bar
            b1_last = b1_bar

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dic = {
                "W1_states": W1_last,
                "b1_states": b1_last,
            }
            return last_param_dic, XQW_mini_batch

        init_params_dic = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
        }
        inputs = {
            "XQ": q.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim),
            "XK": k.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim),
            "XV": v.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim),
            "ttt_lr_eta": ttt_lr_eta.reshape(B, num_heads, num_mini_batch, mini_batch_size, 1),
        }

        # [B,num_heads, num_mini_batch, mini_batch_size, f] -> [num_mini_batch, B, num_heads, mini_batch_size, f]
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, num_heads, mini_batch_size, head_dim), device=device, dtype=dtype
        )
        # [num_mini_batch, B, num_heads, mini_batch_size, f]
        _, XQW_batch = scan(
            compute_mini_batch,
            init_params_dic,
            inputs,
            XQW_batch,
        )

        XQW_batch = XQW_batch.permute(1, 2, 0, 3, 4).reshape(B, num_heads, N, head_dim)

        return XQW_batch

class EvaM2Primal(EvaBasePrimal):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # inner loop weight matrix initialization for TTT-MLP
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, q, k, v, ttt_lr_eta):
        """
        Args:
            q: [B, num_heads, N, head_dim]
            k: [B, num_heads, N, head_dim]
            v: [B, num_heads, N, head_dim]
            ttt_lr_eta: [B, num_heads, N, 1]
        """

        B, num_heads, N, head_dim = q.shape
        num_mini_batch = self.num_mini_batch
        mini_batch_size = N // num_mini_batch
        device = q.device
        dtype = q.dtype


        def compute_mini_batch(params_dic, inputs):
            # [B, nh, f, 4f], nh=num_heads, f=head_dim
            W1_init = params_dic["W1_states"]
            # [B, nh, 1, 4f]
            b1_init = params_dic["b1_states"]
            # [B, nh, 4f, f]
            W2_init = params_dic["W2_states"]
            # [B, nh, 1, f]
            b2_init = params_dic["b2_states"]

            # [B,nh,K,f], K=mini_batch_size
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            X2 = F.gelu(Z1, approximate="tanh")
            Z2 = X2 @ W2_init + b2_init

            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(num_heads, 1, head_dim)
            ln_bias = self.ttt_norm_bias.reshape(num_heads, 1, head_dim)
            # [B,nh,K,f]
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
            # [B, nh, K, 4f]
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

            grad_l_wrt_Z2 = grad_l_wrt_Z2 * ttt_lr_eta_mini_batch / mini_batch_size
            grad_l_wrt_Z1 = grad_l_wrt_Z1 * ttt_lr_eta_mini_batch / mini_batch_size

            # [B, nh, 4f, f]
            grad_W2 = X2.transpose(-2, -1) @ grad_l_wrt_Z2
            # [B, nh, 1, f]
            grad_b2 = torch.sum(grad_l_wrt_Z2, dim=2, keepdim=True)

            # [B, nh, f, 4f]
            grad_W1 = X1.transpose(-2, -1) @ grad_l_wrt_Z1
            # [B, nh, 1, 4f]
            grad_b1 = torch.sum(grad_l_wrt_Z1, dim=2, keepdim=True)

            W2_bar = W2_init - grad_W2
            b2_bar = b2_init - grad_b2
            W1_bar = W1_init - grad_W1
            b1_bar = b1_init - grad_b1

            # [B, nh, K, f] @ [B, nh, f, f]
            Z1_bar = XQ_mini_batch @ W1_bar + b1_bar
            X2_bar = F.gelu(Z1_bar, approximate="tanh")
            Z2_bar = X2_bar @ W2_bar + b2_bar

            W1_last = W1_bar
            b1_last = b1_bar
            W2_last = W2_bar
            b2_last = b2_bar

            Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z2_bar

            last_param_dic = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
            }
            return last_param_dic, XQW_mini_batch

        init_params_dic = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
        }
        inputs = {
            "XQ": q.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim),
            "XK": k.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim),
            "XV": v.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim),
            "ttt_lr_eta": ttt_lr_eta.reshape(B, num_heads, num_mini_batch, mini_batch_size, 1),
        }

        # [B,num_heads, num_mini_batch, mini_batch_size, f] -> [num_mini_batch, B, num_heads, mini_batch_size, f]
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, num_heads, mini_batch_size, head_dim), device=device, dtype=dtype
        )
        # [num_mini_batch, B, num_heads, mini_batch_size, f]
        _, XQW_batch = scan(
            compute_mini_batch,
            init_params_dic,
            inputs,
            XQW_batch,
        )

        XQW_batch = XQW_batch.permute(1, 2, 0, 3, 4).reshape(B, num_heads, N, head_dim)

        return XQW_batch
    


class EvaLinearAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        # attn = (q @ k.transpose(-2, -1))
        # attn = self.attn_drop(attn)
        # x = attn @ v
        # [B, num_heads, f, f]
        kv = k.transpose(-2, -1) @ v

        # [B, num_heads, N, f]
        x = q @ kv / v.shape[-1] / v.shape[-2]

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
            inner_net: str = 'linear_attention',
            ttt_base_lr: float = 1.0,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        if inner_net == 'linear_attention':
            self.attn = EvaLinearAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
            )
        elif inner_net == 'm1_primal':
            self.attn = EvaM1Primal(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
                ttt_base_lr=ttt_base_lr,
            )
        elif inner_net == 'm2_primal':
            self.attn = EvaM2Primal(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
                ttt_base_lr=ttt_base_lr,
            )
        elif inner_net == 'm1_wrapper':
            self.attn = TTTWrapper(
                dim,
                inner_model=InnerLinear(dim, num_heads),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
                ttt_base_lr=ttt_base_lr,
            )
        elif inner_net == 'attn_wrapper':
            self.attn = TTTWrapper(
                dim,
                inner_model=InnerAttention(dim, num_heads),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
                ttt_base_lr=ttt_base_lr,
            )
        else:
            raise ValueError(f"Unsupported inner_net: {inner_net}")
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
            inner_net: str = 'linear_attention',
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        if inner_net == 'linear_attention':
            self.attn = EvaLinearAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
            )
        elif inner_net == 'm1_primal':
            self.attn = EvaM1Primal(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
            )
        elif inner_net == 'm2_primal':
            self.attn = EvaM2Primal(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer if scale_attn_inner else None,
            )
        else:
            raise ValueError(f"Unsupported inner_net: {inner_net}")
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class EvaTTT(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNormFp32,
            init_values: Optional[float] = None,
            class_token: bool = True,
            num_reg_tokens: int = 0,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            inner_net: str = 'linear_attention',
            ttt_base_lr: float = 1.0,
    ):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            num_classes:
            global_pool:
            embed_dim:
            depth:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            drop_rate:
            pos_drop_rate:
            proj_drop_rate:
            attn_drop_rate:
            drop_path_rate:
            norm_layer:
            init_values:
            class_token:
            use_abs_pos_emb:
            use_rot_pos_emb:
            use_post_norm:
            ref_feat_shape:
            head_init_scale:
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                num_prefix_tokens=self.num_prefix_tokens,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                inner_net=inner_net,
                ttt_base_lr=ttt_base_lr,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if pos_embed is not None:
            x = x + pos_embed

        if self.reg_token is not None:
            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            x = torch.cat(to_cat + [x], dim=1)

        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        return x, rot_pos_embed

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int], Tuple[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.
        Args:
            x: Input image tensor
            indices: Take last n blocks if an int, if is a sequence, select by matching indices
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format for EVA-ViT features must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x, rope=rot_pos_embed)
            if i in take_indices:
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int], Tuple[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed, use_reentrant=False)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict


def _create_eva_ttt(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        EvaTTT, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


@register_model
def vit_la_small_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='linear_attention',
    )
    model = _create_eva_ttt('vit_la_small_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_m1_small_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='m1_primal',
    )
    model = _create_eva_ttt('vit_m1_small_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_m2_small_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='m2_primal',
    )
    model = _create_eva_ttt('vit_m2_small_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_la_base_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='linear_attention',
    )
    model = _create_eva_ttt('vit_la_base_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_m1_base_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='m1_primal',
    )
    model = _create_eva_ttt('vit_m1_base_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_m2_base_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='m2_primal',
    )
    model = _create_eva_ttt('vit_m2_base_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_ttt_linear_base_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='m1_wrapper',
    )
    model = _create_eva_ttt('vit_ttt_linear_base_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_ttt_attn_base_patch16_rope_gap_224(pretrained=False, **kwargs) -> EvaTTT:
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        inner_net='attn_wrapper',
    )
    model = _create_eva_ttt('vit_ttt_attn_base_patch16_rope_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model