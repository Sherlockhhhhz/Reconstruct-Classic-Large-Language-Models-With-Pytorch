import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    """
    LLaMA 模型的超参数配置。

    所有字段均对应论文 "LLaMA: Open and Efficient Foundation Language Models"
    以及 Meta 官方实现中的命名习惯。

    参数
    ----
    dim : int
        模型的隐藏维度（即 token embedding 的维度，也是每一层 Transformer
        输入/输出的特征维度）。LLaMA-7B 为 4096，LLaMA-65B 为 8192。

    n_layers : int
        Transformer 解码器块（TransformerBlock）的堆叠层数。
        LLaMA-7B 为 32 层，LLaMA-65B 为 80 层。

    n_heads : int
        多头注意力（Multi-Head Attention）中 Query 的头数。
        每个头的维度 = dim // n_heads。

    n_kv_heads : Optional[int]
        Key/Value 的头数，用于分组查询注意力（Grouped-Query Attention, GQA）。
        - None（默认）：退化为标准多头注意力，KV 头数 == Q 头数。
        - 小于 n_heads 的整数：GQA 模式，多个 Q 头共享同一组 KV 头，
          可显著降低 KV Cache 显存占用（LLaMA-2 70B 使用此特性）。
        - 需满足 n_heads % n_kv_heads == 0。

    vocab_size : int
        词表大小。使用前必须由 tokenizer 填入（默认 -1 为占位符）。
        LLaMA 原版词表大小为 32000。

    multiple_of : int
        FFN 隐藏层维度必须是该值的整数倍，便于硬件（如 Tensor Core）
        对矩阵乘法进行内存对齐优化。默认 256。

    ffn_dim_multiplier : Optional[float]
        对 FFN 隐藏层维度的额外缩放因子。None 表示使用默认计算公式
        （即 SwiGLU 标准公式：2/3 × 4×dim，再对齐到 multiple_of）。
        某些变体（如 LLaMA-2）通过此参数微调 FFN 宽度。

    norm_eps : float
        RMSNorm 中防止除零的数值稳定项 ε。默认 1e-5。

    max_batch_size : int
        KV Cache 预分配时的最大批次大小。推理时 batch size 不能超过此值。

    max_seq_len : int
        KV Cache 预分配时的最大序列长度，同时也是 RoPE 频率预计算的上限。
        LLaMA-1 默认 2048，LLaMA-2 默认 4096。
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization（RMS 归一化）。

    相较于 LayerNorm，RMSNorm 去掉了均值中心化步骤，只做尺度归一化：
        RMSNorm(x) = x / RMS(x) * weight
        其中 RMS(x) = sqrt( mean(x²) + ε )

    这样做计算量更小，且在 LLM 中效果与 LayerNorm 相当。

    参数
    ----
    dim : int
        输入特征的维度，等于 ModelArgs.dim。

    eps : float
        数值稳定项，防止 RMS 为零时出现除零错误。默认 1e-6。

    属性
    ----
    weight : nn.Parameter, shape (dim,)
        可学习的逐元素缩放参数（初始化为全 1），等价于 LayerNorm 的 γ。
        LLaMA 没有偏置项（β）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : Tensor, shape (..., dim)
            任意批次维度 + 特征维度的输入张量。

        返回
        ----
        Tensor, shape (..., dim)
            归一化并缩放后的结果。
        """
        # torch.rsqrt(y) = 1 / sqrt(y)，比先 sqrt 再取倒数数值更稳定
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ──────────────────────────────────────────────
# Rotary Position Embedding（RoPE）相关函数
# ──────────────────────────────────────────────

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 所需的复数旋转因子（cos + i·sin 形式）。

    RoPE 的核心思想：将位置信息编码为旋转矩阵，对 Q/K 向量的每对
    相邻维度 (x_{2i}, x_{2i+1}) 施加角度为 m·θ_i 的旋转，其中：
        θ_i = 1 / (theta^(2i / dim))，i = 0, 1, ..., dim/2 - 1
        m    = token 在序列中的绝对位置

    使用复数表示可将旋转矩阵的 4 次乘法化简为 1 次复数乘法。

    参数
    ----
    dim : int
        每个注意力头的维度（= ModelArgs.dim // ModelArgs.n_heads）。
        只用到 dim//2 个频率，因为每次处理两个实数维度组成一个复数。

    end : int
        预计算的最大位置数，通常为 max_seq_len * 2（留余量）。

    theta : float
        RoPE 基础频率底数，默认 10000（原论文值）。
        值越大，低频分量的波长越长，模型对长序列的外推能力越强。

    返回
    ----
    Tensor, shape (end, dim // 2)，dtype=complex64
        freqs_cis[m, i] = e^{i·m·θ_i} = cos(m·θ_i) + i·sin(m·θ_i)
    """
    # 计算各维度对应的角频率：θ_i = 1 / theta^(2i/dim)，shape (dim//2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # 位置索引：0, 1, ..., end-1，shape (end,)
    t = torch.arange(end, device=freqs.device)
    # 外积得到每个位置、每个频率的角度矩阵，shape (end, dim//2)
    freqs = torch.outer(t, freqs)
    # 转为单位复数 e^{iθ}，模为 1，幅角为 freqs，shape (end, dim//2)，complex64
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    将 freqs_cis 的形状调整为可与 x 做广播乘法的形式。

    参数
    ----
    freqs_cis : Tensor, shape (seq_len, head_dim // 2)
        预计算的旋转因子，complex64。

    x : Tensor, shape (batch, seq_len, n_heads, head_dim // 2)
        已转换为复数的 Q 或 K 张量。

    返回
    ----
    Tensor, shape (1, seq_len, 1, head_dim // 2)
        在 batch 维和 n_heads 维插入大小为 1 的广播维，使乘法自动广播。
    """
    # 只保留第 1 维（seq_len）和最后一维（head_dim//2），其余维设为 1
    shape = [1 if i != 1 and i != x.ndim - 1 else d for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 施加旋转位置编码（RoPE）。

    实现原理：将每个头的实数向量最后一维两两配对，视作复数，
    再与旋转因子 e^{imθ} 相乘，等价于对每对维度施加旋转矩阵。

    参数
    ----
    xq : Tensor, shape (batch, seq_len, n_heads, head_dim)
        Query 张量（实数）。

    xk : Tensor, shape (batch, seq_len, n_kv_heads, head_dim)
        Key 张量（实数）。

    freqs_cis : Tensor, shape (seq_len, head_dim // 2)，complex64
        当前位置窗口对应的旋转因子。

    返回
    ----
    xq_out : Tensor, shape (batch, seq_len, n_heads, head_dim)
    xk_out : Tensor, shape (batch, seq_len, n_kv_heads, head_dim)
        施加 RoPE 后的 Q 和 K（dtype 与输入相同）。
    """
    # 将实数 head_dim 维度重排为 (head_dim//2, 2) 后解释为复数
    # shape: (batch, seq_len, n_heads, head_dim//2)，complex64
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整 freqs_cis 形状以支持广播：(1, seq_len, 1, head_dim//2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 复数乘法 = 旋转；flatten(3) 将最后两维 (head_dim//2, 2) 还原为 head_dim
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 转回原始 dtype（如 bfloat16），保持与上下文一致
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 头沿 n_kv_heads 维度重复 n_rep 次，使其头数与 Q 对齐。

    这是 GQA（Grouped-Query Attention）到标准 MHA 的展开操作：
    每组 n_rep 个 Q 头共享同一个 KV 头，展开后即可复用标准 SDPA。

    参数
    ----
    x : Tensor, shape (batch, seq_len, n_kv_heads, head_dim)
        Key 或 Value 张量。

    n_rep : int
        重复次数 = n_heads // n_kv_heads。
        为 1 时直接返回原张量（退化为 MHA，无需复制）。

    返回
    ----
    Tensor, shape (batch, seq_len, n_kv_heads * n_rep, head_dim)
        展开后的 Key 或 Value，头数与 Q 一致。
    """
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 在 n_kv_heads 维度后插入大小为 n_rep 的新维度，expand 不复制内存，
    # reshape 触发实际内存展开
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    分组查询注意力（Grouped-Query Attention, GQA）模块，带 KV Cache。

    当 n_kv_heads == n_heads 时退化为标准多头注意力（MHA）。
    当 n_kv_heads == 1 时退化为多查询注意力（MQA）。

    注意力计算公式：
        Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V

    参数（构造函数）
    ----------------
    args : ModelArgs
        模型配置，从中读取以下字段：
        - args.dim         : 模型隐藏维度，作为 wq/wk/wv 的输入维度
        - args.n_heads     : Q 的头数
        - args.n_kv_heads  : KV 的头数（None 则等于 n_heads）
        - args.max_batch_size : KV Cache 的 batch 维预分配大小
        - args.max_seq_len    : KV Cache 的序列维预分配大小

    属性
    ----
    n_heads : int
        Query 的头数。

    n_kv_heads : int
        Key/Value 的头数（已将 None 转换为 n_heads）。

    n_rep : int
        每个 KV 头对应的 Q 头数量（= n_heads // n_kv_heads）。
        GQA 展开时的重复因子。

    head_dim : int
        每个注意力头的维度（= dim // n_heads）。

    wq : nn.Linear(dim, n_heads * head_dim, bias=False)
        Query 投影矩阵。

    wk : nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        Key 投影矩阵。KV 头数更少，参数量小于 wq。

    wv : nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        Value 投影矩阵。

    wo : nn.Linear(n_heads * head_dim, dim, bias=False)
        输出投影矩阵，将所有头的拼接结果映射回 dim 维度。

    cache_k : Tensor, shape (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        Key 的 KV Cache，推理时避免重复计算历史 token 的 K。

    cache_v : Tensor, shape (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        Value 的 KV Cache。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 预分配 KV Cache，推理时按 start_pos 写入，避免每步重新分配
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        参数
        ----
        x : Tensor, shape (batch, seq_len, dim)
            当前步输入的隐藏状态。
            - prefill 阶段：seq_len 为完整输入序列长度，start_pos = 0。
            - decode 阶段：seq_len = 1（逐 token 生成），start_pos 为已生成长度。

        start_pos : int
            当前 token 在整体序列中的起始位置索引，用于：
            1. 从 freqs_cis 中截取对应位置的旋转因子。
            2. 确定向 KV Cache 写入的槽位。
            3. 从 KV Cache 读取历史 KV 的长度（0 ~ start_pos + seq_len）。

        freqs_cis : Tensor, shape (seq_len, head_dim // 2)，complex64
            当前位置窗口对应的 RoPE 旋转因子（由 Transformer.forward 切片传入）。

        mask : Optional[Tensor], shape (seq_len, start_pos + seq_len)
            因果注意力掩码。
            - decode 阶段（seq_len=1）：无需掩码，传 None。
            - prefill 阶段：上三角为 -inf、下三角及对角为 0 的矩阵，
              同时在左侧拼接全零列对应已缓存的前缀 token。

        返回
        ----
        Tensor, shape (batch, seq_len, dim)
            注意力输出，经 wo 投影后与输入维度相同。
        """
        bsz, seq_len, _ = x.shape

        # ── 线性投影 ──────────────────────────────────────────────
        # xq: (bsz, seq_len, n_heads,    head_dim)
        # xk: (bsz, seq_len, n_kv_heads, head_dim)
        # xv: (bsz, seq_len, n_kv_heads, head_dim)
        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # ── 施加旋转位置编码（仅 Q 和 K 需要，V 不需要）──────────
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # ── 更新 KV Cache ─────────────────────────────────────────
        # 将 cache 移到与 xq 相同的设备/dtype（首次推理时做一次迁移）
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos : start_pos + seq_len] = xk
        self.cache_v[:bsz, start_pos : start_pos + seq_len] = xv

        # ── 读取完整历史 KV（包含本步新写入的部分）──────────────
        keys   = self.cache_k[:bsz, : start_pos + seq_len]   # (bsz, total_len, n_kv_heads, head_dim)
        values = self.cache_v[:bsz, : start_pos + seq_len]   # (bsz, total_len, n_kv_heads, head_dim)

        # ── GQA 展开：将 KV 头数复制至与 Q 头数一致 ─────────────
        keys   = repeat_kv(keys,   self.n_rep)   # (bsz, total_len, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)   # (bsz, total_len, n_heads, head_dim)

        # ── Scaled Dot-Product Attention ──────────────────────────
        # 转置到 (bsz, n_heads, seq/total_len, head_dim) 以便 matmul
        xq    = xq.transpose(1, 2)       # (bsz, n_heads, seq_len,   head_dim)
        keys   = keys.transpose(1, 2)    # (bsz, n_heads, total_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, total_len, head_dim)

        # scores: (bsz, n_heads, seq_len, total_len)
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / scale

        if mask is not None:
            scores = scores + mask  # 广播到所有 batch 和 head

        # 在 float32 下做 softmax 保证数值稳定，再转回原 dtype
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # out: (bsz, n_heads, seq_len, head_dim) -> (bsz, seq_len, n_heads * head_dim)
        out = torch.matmul(scores, values)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # 输出投影：将多头拼接结果映射回 dimf
        return self.wo(out)


class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络（Feed-Forward Network）。

    LLaMA 使用 SwiGLU 替代原始 Transformer 的 ReLU FFN：
        FFN_SwiGLU(x) = W2 · ( SiLU(W1·x)  ⊙  W3·x )

    其中：
        - SiLU(x) = x · sigmoid(x)（也称 Swish-1）
        - ⊙ 表示逐元素乘法（门控机制）
        - W1 为"门"路径，W3 为"值"路径，W2 为输出投影

    SwiGLU 相比 ReLU 在同等参数量下效果更好，但由于多了 W3 矩阵，
    为保持总参数量不变，隐藏层维度需从 4×dim 缩小到 (2/3)×4×dim ≈ 2.67×dim。

    参数（构造函数）
    ----------------
    dim : int
        输入和输出的维度（= ModelArgs.dim）。

    hidden_dim : int
        FFN 隐藏层的基础维度，传入前通常为 4 * dim。
        构造函数内部会对其进行以下调整：
        1. 乘以 2/3（SwiGLU 参数量补偿）
        2. 乘以 ffn_dim_multiplier（若不为 None）
        3. 向上取整到 multiple_of 的最小倍数（内存对齐）

    multiple_of : int
        最终隐藏层维度对齐的粒度，确保矩阵维度对硬件友好。

    ffn_dim_multiplier : Optional[float]
        隐藏层维度的额外缩放因子，用于模型变体调整。None 表示不额外缩放。

    属性
    ----
    w1 : nn.Linear(dim, hidden_dim, bias=False)
        门路径的投影矩阵，输出经 SiLU 激活后作为门控信号。

    w2 : nn.Linear(hidden_dim, dim, bias=False)
        输出投影矩阵，将门控后的隐藏表示投影回 dim。

    w3 : nn.Linear(dim, hidden_dim, bias=False)
        值路径的投影矩阵，输出与 SiLU(w1·x) 逐元素相乘。
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # SwiGLU 参数量补偿：用 2/3 倍抵消第三个矩阵 w3 引入的额外参数
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # 向上对齐到 multiple_of 的倍数，公式：ceil(x/m) * m
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : Tensor, shape (batch, seq_len, dim)

        返回
        ----
        Tensor, shape (batch, seq_len, dim)
            SwiGLU 前馈输出。
        """
        # SiLU(w1·x) ⊙ (w3·x)，再经 w2 投影
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    单个 LLaMA Transformer 解码器块（Pre-Norm 结构）。

    结构（与原始 GPT/BERT 的 Post-Norm 不同，LLaMA 使用 Pre-Norm）：
        h   = x + Attention( RMSNorm(x) )
        out = h + FFN( RMSNorm(h) )

    Pre-Norm 将归一化放在子层输入处，训练更稳定，梯度流更顺畅。

    参数（构造函数）
    ----------------
    layer_id : int
        当前块在整个网络中的层编号（0-indexed），目前仅作为标识存储，
        可用于调试或实现层级学习率缩放等技巧。

    args : ModelArgs
        模型配置，用于实例化子模块。

    属性
    ----
    attention : Attention
        多头/分组查询注意力模块。

    feed_forward : FeedForward
        SwiGLU 前馈网络。

    attention_norm : RMSNorm
        注意力子层前的归一化，作用于输入 x。

    ffn_norm : RMSNorm
        FFN 子层前的归一化，作用于注意力输出 h。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,      # 基础隐藏维度，内部会调整
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm       = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        参数
        ----
        x : Tensor, shape (batch, seq_len, dim)
            上一层（或 token embedding）的输出。

        start_pos : int
            当前 token 的起始位置，透传给 Attention 用于 KV Cache 和 RoPE 切片。

        freqs_cis : Tensor, shape (seq_len, head_dim // 2)，complex64
            当前位置窗口的 RoPE 旋转因子，透传给 Attention。

        mask : Optional[Tensor]
            因果掩码，透传给 Attention。

        返回
        ----
        Tensor, shape (batch, seq_len, dim)
            经注意力 + FFN 残差连接后的输出。
        """
        # Pre-Norm 残差：先归一化再做注意力，结果加回输入
        h   = x   + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        # Pre-Norm 残差：先归一化再做 FFN，结果加回
        out = h   + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    LLaMA 完整的自回归 Transformer 语言模型（仅解码器）。

    模型结构：
        tokens
          └─ tok_embeddings          # token embedding 查表
               └─ layers[0..N-1]    # N 个 TransformerBlock（Pre-Norm）
                    └─ norm          # 最终 RMSNorm
                         └─ output  # lm_head：投影到词表大小

    与 GPT-2 的主要区别：
        - 位置编码：RoPE（相对位置，无可学习参数）取代绝对位置编码
        - 归一化：RMSNorm（Pre-Norm）取代 LayerNorm（Post-Norm）
        - 激活函数：SwiGLU 取代 GELU
        - 注意力：支持 GQA，带 KV Cache

    参数（构造函数）
    ----------------
    params : ModelArgs
        完整的模型超参数配置。

    属性
    ----
    tok_embeddings : nn.Embedding(vocab_size, dim)
        token ID → dim 维向量的嵌入表。

    layers : nn.ModuleList
        由 n_layers 个 TransformerBlock 组成的解码器层列表。

    norm : RMSNorm
        最后一层输出后的归一化，在投影到词表前施加。

    output : nn.Linear(dim, vocab_size, bias=False)
        语言模型头（lm_head），将 dim 维隐藏态投影为词表上的 logits。
        通常与 tok_embeddings.weight 共享权重（weight tying），
        此处为简洁起见未实现绑定。

    freqs_cis : Tensor, shape (max_seq_len * 2, head_dim // 2)，complex64
        预计算并缓存的 RoPE 旋转因子，推理时按位置切片。
        预计算长度为 max_seq_len * 2，为未来可能的序列长度扩展留余量。
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params    = params
        self.vocab_size = params.vocab_size
        self.n_layers  = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, params) for i in range(params.n_layers)]
        )
        self.norm   = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 预计算 RoPE 旋转因子并注册为 buffer（不参与梯度更新，随模型保存/加载）
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                params.dim // params.n_heads,   # 每个头的维度
                params.max_seq_len * 2,          # 预计算长度留余量
            ),
            persistent=False,   # 不写入 state_dict，每次从公式重新生成
        )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        参数
        ----
        tokens : Tensor, shape (batch, seq_len)，dtype=torch.long
            当前步输入的 token ID 序列。
            - prefill（首次输入完整 prompt）：seq_len 为 prompt 长度，start_pos = 0。
            - decode（逐步生成）：seq_len = 1，start_pos 为已生成的 token 数。

        start_pos : int
            tokens 在完整序列中的起始位置偏移，用于从预计算的 freqs_cis
            中截取正确位置段，以及指导 KV Cache 的读写。

        返回
        ----
        logits : Tensor, shape (batch, seq_len, vocab_size)，dtype=float32
            每个位置上词表各 token 的未归一化对数概率（logits）。
            取 [:, -1, :] 即可得到下一个 token 的预测分布。
        """
        _, seq_len = tokens.shape
        # token ID → 连续 embedding 向量，shape: (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # 从预计算缓存中截取当前位置段的旋转因子
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(h.device)

        # ── 构造因果掩码 ──────────────────────────────────────────
        # decode 阶段 seq_len=1，每个 token 只与自身和历史 KV 计算，无需掩码
        mask = None
        if seq_len > 1:
            # 上三角填 -inf：当前 token 不能看到其右侧的 token
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # 左侧拼接 start_pos 列的全零：允许关注所有已缓存的历史 KV
            # 最终 shape: (seq_len, start_pos + seq_len)
            mask = torch.hstack([
                torch.zeros((seq_len, start_pos), device=tokens.device),
                mask,
            ]).type_as(h)

        # ── 逐层前向传播 ──────────────────────────────────────────
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # 最终归一化 + lm_head 投影
        h      = self.norm(h)
        logits = self.output(h).float()   # 保持 float32 精度用于采样/损失计算
        return logits
    

if __name__ == "__main__":
    cfg = ModelArgs()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model = Transformer(cfg).to(device)
    print(model)
    print(f'模型参数量为{sum(p.numel() for p in model.parameters())}')