import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）。

    将输入的 Q/K/V 分别投影到 num_heads 个子空间，在每个子空间独立
    计算缩放点积注意力，最后拼接并投影回 d_model 维度。
    多头机制让模型能同时关注序列不同位置的不同语义特征。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model:    模型的统一向量维度，也是 Q/K/V 投影的输入输出维度。
                        必须能被 num_heads 整除，因为每个头的维度 d_k = d_model // num_heads。
            num_heads:  注意力头的数量。头数越多，模型能捕捉的特征子空间越丰富，
                        但每头的维度 d_k 随之减小。
            dropout:    注意力权重的 dropout 概率，用于训练时随机屏蔽注意力连接，
                        防止过拟合。默认 0.1。
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个注意力头的维度

        # 将输入线性投影到 Q / K / V 空间（维度保持 d_model，由 split_head 再拆分多头）
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影矩阵
        self.W_k = nn.Linear(d_model, d_model)  # Key   投影矩阵
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影矩阵
        self.W_o = nn.Linear(d_model, d_model)  # 多头拼接后的输出投影矩阵

        self.dropout = nn.Dropout(dropout)

    def split_head(self, x):
        """
        将最后一维 d_model 拆分为 (num_heads, d_k) 并转置，使头的维度提前。

        (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k)
        """
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力：score = softmax(Q·Kᵀ / √d_k) · V

        Args:
            Q:    Query 矩阵，形状 (batch, heads, seq_len, d_k)
            K:    Key   矩阵，形状 (batch, heads, seq_len, d_k)
            V:    Value 矩阵，形状 (batch, heads, seq_len, d_k)
            mask: 注意力掩码，0 的位置会被置为 -inf（softmax 后趋近 0），
                  形状为 (batch, 1, 1, src_len) 或 (batch, 1, tgt_len, tgt_len)。

        Returns:
            output:       加权求和后的上下文向量，形状与 V 相同。
            attn_weights: 注意力权重矩阵，可用于可视化。
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V), attn_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: 查询序列，形状 (batch, seq_len_q, d_model)
            key:   键序列，  形状 (batch, seq_len_k, d_model)
            value: 值序列，  形状 (batch, seq_len_v, d_model)，seq_len_k == seq_len_v
            mask:  注意力掩码，屏蔽 padding 或未来位置

        Returns:
            output:       输出张量，形状 (batch, seq_len_q, d_model)
            attn_weights: 注意力权重，形状 (batch, heads, seq_len_q, seq_len_k)
        """
        Q = self.split_head(self.W_q(query))
        K = self.split_head(self.W_k(key))
        V = self.split_head(self.W_v(value))

        x, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头：(batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(x), attn_weights


class FeedForward(nn.Module):
    """
    逐位置前馈网络（Position-wise Feed-Forward Network）。

    对序列中每个位置独立地执行两层全连接变换：
        FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
    中间维度 d_ff 通常是 d_model 的 4 倍，用于增强模型的非线性表达能力。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model:  输入/输出的特征维度，与模型整体保持一致。
            d_ff:     中间隐藏层的维度，通常取 4 * d_model（原论文为 2048）。
                      更大的 d_ff 提升模型容量，但也增加计算量和参数量。
            dropout:  在第一层激活之后应用 dropout，防止过拟合。默认 0.1。
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # 升维：d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)   # 降维：d_ff -> d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码（Sinusoidal Positional Encoding）。

    Transformer 本身没有循环或卷积结构，无法感知序列顺序，
    因此需要向嵌入向量中注入位置信息。编码公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    使用固定编码（非可学习），好处是推理时可以外推到训练集未见过的序列长度。
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model:  位置编码的维度，必须与词嵌入维度相同，才能直接相加。
            max_len:  预计算的最大序列长度。推理序列超过该长度时会越界，
                      通常设为远大于实际最大长度的值（默认 5000）。
            dropout:  将位置编码叠加到词嵌入后施加 dropout，防止过拟合。默认 0.1。
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵并注册为 buffer（不参与梯度更新，但随模型保存/迁移设备）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # 对应公式中的 1/10000^(2i/d_model)，用指数形式避免数值溢出
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，便于与 batch 广播
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    Transformer 编码器的单层结构。

    每层包含两个子层，均使用残差连接（Add）和层归一化（Norm）：
        1. 多头自注意力（Self-Attention）：让序列中每个位置关注同一序列的所有位置。
        2. 逐位置前馈网络（FFN）：对每个位置独立地做非线性变换。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model:   模型的统一向量维度，贯穿整个编码器。
            num_heads: 多头注意力的头数，需能整除 d_model。
            d_ff:      前馈网络中间层的维度，通常为 4 * d_model。
            dropout:   用于注意力权重和子层输出的 dropout 概率。默认 0.1。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 自注意力子层
        self.ff = FeedForward(d_model, d_ff, dropout)                     # 前馈子层
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力之后的层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 前馈网络之后的层归一化
        self.dropout = nn.Dropout(dropout)  # 残差连接前对子层输出施加的 dropout

    def forward(self, x, src_mask=None):
        """
        Args:
            x:        编码器输入，形状 (batch, src_len, d_model)
            src_mask: 源序列的 padding 掩码，形状 (batch, 1, 1, src_len)，
                      屏蔽 padding token，使其不被注意到。

        Returns:
            x: 经过自注意力和 FFN 变换后的输出，形状与输入相同。
        """
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))   # 残差 + 层归一化
        x = self.norm2(x + self.dropout(self.ff(x))) # 残差 + 层归一化
        return x


class DecoderLayer(nn.Module):
    """
    Transformer 解码器的单层结构。

    每层包含三个子层，均使用残差连接和层归一化：
        1. 掩码多头自注意力（Masked Self-Attention）：解码器对已生成 token 的自注意力，
           causal mask 屏蔽未来位置，保证自回归生成的因果性。
        2. 编码器-解码器交叉注意力（Cross-Attention）：以解码器状态为 Query，
           编码器输出为 Key/Value，让解码器关注源序列信息。
        3. 逐位置前馈网络（FFN）：同编码器层。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model:   模型的统一向量维度。
            num_heads: 注意力头数，需能整除 d_model。
            d_ff:      前馈网络中间层维度，通常为 4 * d_model。
            dropout:   dropout 概率。默认 0.1。
        """
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)  # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 交叉注意力
        self.ff         = FeedForward(d_model, d_ff, dropout)              # 前馈子层
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力之后的层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力之后的层归一化
        self.norm3 = nn.LayerNorm(d_model)  # 前馈网络之后的层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x:          解码器输入，形状 (batch, tgt_len, d_model)
            enc_output: 编码器最终输出，形状 (batch, src_len, d_model)，
                        作为交叉注意力的 Key 和 Value。
            src_mask:   源序列 padding 掩码，形状 (batch, 1, 1, src_len)，
                        用于交叉注意力，屏蔽编码器端的 padding token。
            tgt_mask:   目标序列掩码，形状 (batch, 1, tgt_len, tgt_len)，
                        同时屏蔽 padding 和未来位置（causal mask）。

        Returns:
            x: 经过三个子层变换后的输出，形状 (batch, tgt_len, d_model)
        """
        # 子层 1：掩码自注意力
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 子层 2：交叉注意力（Q 来自解码器，K/V 来自编码器输出）
        attn_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_out))

        # 子层 3：前馈网络
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Transformer(nn.Module):
    """
    标准 Encoder-Decoder Transformer（Vaswani et al., 2017 "Attention Is All You Need"）。

    整体流程：
        1. 源/目标序列经词嵌入后乘以 √d_model 进行缩放，再叠加位置编码。
        2. 编码器由 num_encoder_layers 个 EncoderLayer 堆叠，输出源序列的上下文表示。
        3. 解码器由 num_decoder_layers 个 DecoderLayer 堆叠，自回归地生成目标序列。
        4. 最终通过线性层将解码器输出映射到目标词表，得到每步的 logits。
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        pad_idx=0,
    ):
        """
        Args:
            src_vocab_size:      源语言词表大小，决定源端嵌入矩阵的行数。
            tgt_vocab_size:      目标语言词表大小，决定目标端嵌入矩阵和输出线性层的维度。
            d_model:             模型统一的向量维度，贯穿嵌入、编解码器及位置编码。
                                 原论文取 512；较小任务可降至 128/256 以节省资源。
            num_heads:           多头注意力的头数，需能整除 d_model。原论文取 8。
            num_encoder_layers:  编码器堆叠的层数。原论文取 6；层数越多表达能力越强，
                                 但训练更难、推理更慢。
            num_decoder_layers:  解码器堆叠的层数。原论文取 6。
            d_ff:                每个编/解码器层中前馈网络的中间维度。原论文取 2048，
                                 通常为 4 * d_model。
            max_len:             位置编码预计算的最大序列长度，需大于实际最长序列。默认 5000。
            dropout:             全局 dropout 概率，应用于嵌入、注意力权重和子层输出。默认 0.1。
            pad_idx:             padding token 的索引（通常为 0），用于：
                                 1. 嵌入层忽略 padding（padding_idx）；
                                 2. 构造注意力掩码时屏蔽 padding 位置。
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # 词嵌入层：将 token id 转换为 d_model 维稠密向量
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        # 位置编码（源/目标序列共用同一个位置编码模块）
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 编码器：多层 EncoderLayer 堆叠
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        # 解码器：多层 DecoderLayer 堆叠
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        # 输出层：将解码器隐状态映射到目标词表的 logits
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        # Xavier 均匀初始化所有二维及以上参数（嵌入矩阵、线性层权重），
        # 有助于梯度在深层网络中稳定传播。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        构造源序列的 padding 掩码。

        Args:
            src: 源序列 token id，形状 (batch, src_len)

        Returns:
            掩码张量，形状 (batch, 1, 1, src_len)，非 padding 位置为 True。
            广播后可覆盖 (batch, heads, query_len, src_len) 的注意力分数矩阵。
        """
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        """
        构造目标序列的掩码（padding mask 与 causal mask 的交集）。

        Args:
            tgt: 目标序列 token id，形状 (batch, tgt_len)

        Returns:
            掩码张量，形状 (batch, 1, tgt_len, tgt_len)。
            同时屏蔽：
              - padding token（padding mask）
              - 当前位置之后的 token（下三角 causal mask），保证解码器的自回归性质。
        """
        tgt_len = tgt.size(1)
        pad_mask   = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)         # (batch, 1, 1, tgt_len)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()  # (tgt_len, tgt_len)
        return pad_mask & causal_mask

    def encode(self, src, src_mask):
        """
        对源序列执行完整的编码过程。

        Args:
            src:      源序列 token id，形状 (batch, src_len)
            src_mask: 源序列 padding 掩码，形状 (batch, 1, 1, src_len)

        Returns:
            编码器最终输出，形状 (batch, src_len, d_model)
        """
        # 词嵌入乘以 √d_model 进行缩放，使嵌入幅度与位置编码处于相近量级
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """
        对目标序列执行完整的解码过程。

        Args:
            tgt:        目标序列 token id，形状 (batch, tgt_len)
            enc_output: 编码器输出，形状 (batch, src_len, d_model)
            src_mask:   源序列 padding 掩码，供交叉注意力使用
            tgt_mask:   目标序列掩码（padding + causal）

        Returns:
            解码器最终输出，形状 (batch, tgt_len, d_model)
        """
        x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        """
        完整的 Transformer 前向传播（训练阶段使用 teacher forcing）。

        Args:
            src: 源序列 token id，形状 (batch, src_len)
            tgt: 目标序列 token id，形状 (batch, tgt_len)，
                 训练时为目标序列去掉最后一个 token（右移一位输入解码器）。

        Returns:
            logits，形状 (batch, tgt_len, tgt_vocab_size)，
            经 softmax 后即为每个位置的词表概率分布。
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)


if __name__ == "__main__":

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 快速验证模型结构
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
    )
    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 8))
    out = model(src, tgt)
    print(f"输出形状: {out.shape}")  # (2, 8, 1000)
    print(f"可训练参数量: {count_parameters(model):,}")
