import math        # 用于 sqrt 等数学运算（注意力缩放因子）
from dataclasses import dataclass   # 轻量级配置类，避免手写 __init__
from typing import Optional         # 类型注解，Optional[X] = X | None

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPT2Config:
    """GPT-2 模型结构超参数（Small 规模，适合本地单卡训练）。
    
    原始 GPT-2 Small：n_layer=12, n_head=12, n_embd=768，这里缩减以降低显存需求。
    """
    vocab_size: int   = 50257   # GPT-2 BPE 词表大小（固定值，与 tiktoken gpt2 编码器一致）
    block_size: int   = 256     # 上下文长度（最大序列长度），原始 GPT-2 为 1024
    n_layer: int      = 6       # Transformer Block 层数
    n_head: int       = 6       # 多头注意力头数；每头维度 = n_embd // n_head = 64
    n_embd: int       = 384     # 嵌入维度（隐层宽度）
    dropout: float    = 0.1     # Dropout 概率，用于 attention 和残差连接处

@dataclass
class TrainConfig:
    """训练过程超参数。"""
    batch_size: int      = 32    # 每个 mini-batch 的样本数
    max_epochs: int      = 5     # 训练总轮数
    lr: float            = 3e-4  # AdamW 峰值学习率
    weight_decay: float  = 0.1   # L2 正则化系数（仅作用于权重矩阵，不含 bias/LN）
    grad_clip: float     = 1.0   # 梯度裁剪阈值，防止梯度爆炸
    eval_interval: int   = 200   # 每训练多少步进行一次验证集评估
    eval_iters: int      = 50    # 评估时使用多少个 batch（越大越准确，但越慢）


# ── 4.1 Causal Self-Attention ──────────────────────────────────────────────
# 因果自注意力：每个位置只能关注自身及其左侧的 token（autoregressive 约束）

class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # n_embd 必须能被 n_head 整除，才能等分给每个注意力头
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.n_embd   = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head  # 每个头的维度

        # 用一个大线性层同时计算 Q / K / V，比三个独立线性层更高效
        # 输出维度 3*n_embd，之后按 n_embd 切分成三份
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # 多头拼接后的输出投影，将 n_embd 映射回 n_embd
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.attn_drop  = nn.Dropout(cfg.dropout)  # 注意力权重上的 dropout
        self.resid_drop = nn.Dropout(cfg.dropout)  # 残差输出上的 dropout

        # 因果掩码：下三角矩阵，形状 (1,1,T,T)，广播到 (B, n_head, T, T)
        # 值为 0 的位置（上三角）在 softmax 前填 -inf，使其注意力权重为 0
        # register_buffer：不作为模型参数，但会随 model.to(device) 一起移动
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
              .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # 批大小、序列长度、嵌入维度

        # ── 步骤1：线性投影并分割 Q / K / V ──
        qkv = self.c_attn(x)                      # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)   # 各 (B, T, C)

        # ── 步骤2：变形为多头格式 (B, n_head, T, head_dim) ──
        # view 将 C 维分成 (n_head, head_dim)，transpose 把 n_head 提到第2维
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── 步骤3：Scaled Dot-Product Attention ──
        # 缩放因子 1/sqrt(d_k) 防止内积过大导致 softmax 梯度消失
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale    # (B, n_head, T, T)

        # 将未来位置（上三角）置为 -inf，softmax 后这些位置权重趋近于 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)               # 沿 key 维度归一化
        att = self.attn_drop(att)

        # ── 步骤4：加权聚合 value，拼接多头 ──
        y = att @ v                                      # (B, n_head, T, head_dim)
        # transpose 还原维度顺序，contiguous 保证内存连续后再 view 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # 输出投影 + 残差 dropout
        return self.resid_drop(self.c_proj(y))
    
# ── 4.2 Feed-Forward (MLP) ────────────────────────────────────────────────
# 位置逐点前馈网络：对每个 token 位置独立地做两层线性变换
# 中间层维度扩大 4 倍（GPT-2 论文设定），相当于"知识存储"空间

class MLP(nn.Module):

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # 升维：n_embd -> 4*n_embd，让网络有更大的特征空间做非线性变换
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        # 降维：4*n_embd -> n_embd，输出与残差连接保持同维度
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop   = nn.Dropout(cfg.dropout)
        # GPT-2 使用 GELU 激活（tanh 近似版），比 ReLU 更平滑，梯度消失更少
        # approximate="tanh" 与原始论文的近似公式一致，在 GPU/MPS 上更快
        self.act    = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 升维 -> 激活 -> 降维 -> dropout，顺序不可颠倒
        return self.drop(self.c_proj(self.act(self.c_fc(x))))
    
# ── 4.3 Transformer Block（Pre-LayerNorm） ─────────────────────────────────
# GPT-2 采用 Pre-LN（层归一化放在子层之前），与原始 Transformer 的 Post-LN 不同。
# Pre-LN 训练更稳定，不容易出现梯度消失/爆炸，适合深层网络。
# 残差连接：x = x + sublayer(LN(x))，保证梯度可以直接流过残差路径。

class Block(nn.Module):

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)   # 注意力子层前的 LayerNorm
        self.attn = CausalSelfAttention(cfg)    # 因果自注意力
        self.ln_2 = nn.LayerNorm(cfg.n_embd)   # MLP 子层前的 LayerNorm
        self.mlp  = MLP(cfg)                    # 前馈网络

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接1：先 LayerNorm，再 Attention，最后加回原始 x
        x = x + self.attn(self.ln_1(x))
        # 残差连接2：先 LayerNorm，再 MLP，最后加回 x
        x = x + self.mlp(self.ln_2(x))
        return x
    
# ── 4.4 GPT-2 整体架构 ────────────────────────────────────────────────────
# 完整的 Decoder-only Transformer：Embedding → N×Block → LN → LM Head

class GPT2(nn.Module):

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg

        # 使用 ModuleDict 组织 Transformer 核心组件，方便按名称访问
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),  # token embedding：将 token id 映射到向量
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),  # position embedding：位置 0~block_size-1
            drop = nn.Dropout(cfg.dropout),                   # embedding 层输出的 dropout
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),  # n_layer 个 Transformer Block
            ln_f = nn.LayerNorm(cfg.n_embd),                  # 最终输出的 LayerNorm（Pre-LN 的尾部归一化）
        ))

        # Language Model Head：将隐层向量映射到词表概率分布
        # bias=False：GPT-2 原始实现无 bias，减少参数量
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight Tying：lm_head 的权重与 token embedding 共享
        # 直觉：同一个 token 的嵌入向量和输出逻辑向量应保持一致的语义
        # 实践效果：显著减少参数量（约 20M），同时提升训练效率和生成质量
        self.transformer.wte.weight = self.lm_head.weight

        self._init_weights()
        print(f"GPT-2 parameters: {self.num_params():,}")

    def _init_weights(self):
        """按照 GPT-2 论文的初始化方案初始化权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 权重用均值 0、标准差 0.02 的正态分布初始化
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)   # bias 初始化为 0
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 残差投影层的特殊缩放：std 除以 sqrt(2 * n_layer)
        # 动机：每层有 2 条残差路径（attention + mlp），深层网络中残差累加会使
        # 方差随层数增长。提前缩小初始化标准差可以抵消这种累积效应，稳定训练。
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))

    def num_params(self) -> int:
        """返回模型总参数量（包含共享权重，会重复计数）。"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        """前向传播，返回 (logits, loss)。

        Args:
            idx:     token id 序列，形状 (B, T)
            targets: 目标序列，形状 (B, T)；为 None 时仅做推理，不计算 loss
        Returns:
            logits: 每个位置对应的词表 logits，形状 (B, T, vocab_size)
            loss:   交叉熵损失标量；targets 为 None 时返回 None
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过 block_size {self.cfg.block_size}"

        # 生成位置索引 [0, 1, ..., T-1]，与 idx 在同一设备上
        pos = torch.arange(T, device=idx.device)          # (T,)

        # token embedding + position embedding，逐元素相加
        tok_emb = self.transformer.wte(idx)               # (B, T, C)
        pos_emb = self.transformer.wpe(pos)               # (T, C)，广播到 (B, T, C)
        x = self.transformer.drop(tok_emb + pos_emb)     # (B, T, C)

        # 顺序通过 n_layer 个 Transformer Block
        for block in self.transformer.h:
            x = block(x)

        # 最终 LayerNorm（Pre-LN 架构在整个堆叠的末尾再做一次归一化）
        x = self.transformer.ln_f(x)

        # 将隐层向量映射到词表大小的 logits
        logits = self.lm_head(x)                          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 展平为 (B*T, vocab_size) 和 (B*T,) 后计算交叉熵
            # 每个位置的 logit 预测对应位置的下一个 token（即 targets[t]）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """自回归文本生成。

        Args:
            idx:            初始 token id 序列，形状 (1, T)（batch=1）
            max_new_tokens: 最多生成多少个新 token
            temperature:    采样温度；< 1 使分布更尖锐（保守），> 1 更平坦（多样）
            top_k:          Top-K 截断采样；只保留概率最高的 k 个 token 后重新归一化
        Returns:
            追加了新生成 token 后的完整序列，形状 (1, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 若序列超过 block_size，只取最后 block_size 个 token（滑动窗口）
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            # 只取最后一个位置的 logit（预测下一个 token）
            logits = logits[:, -1, :] / temperature      # (1, vocab_size)

            if top_k is not None:
                # 找出 top-k 中最小的值，低于该值的 logit 置为 -inf（过滤掉）
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # softmax 得到概率分布，然后多项式采样
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # 将新生成的 token 追加到序列末尾
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

if __name__ == "__main__":
    model_cfg = GPT2Config()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = GPT2(model_cfg).to(device)