import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from util import masked_softmax
import math


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k, self.d_q, self.d_v = d_k, d_k, d_v

        self.heads = []
        for i in range(num_heads): # would be better off avoiding forloop, if only time was there :(
            d = nn.ModuleDict({
                'w_q': nn.Linear(self.hidden_size, self.d_q, bias=False),
                'w_k': nn.Linear(self.hidden_size, self.d_k, bias=False),
                'w_v': nn.Linear(self.hidden_size, self.d_v, bias=False)
            })
            d = d.to('cuda')
            nn.init.xavier_uniform_(d['w_q'].weight)
            nn.init.xavier_uniform_(d['w_k'].weight)
            nn.init.xavier_uniform_(d['w_v'].weight)
            self.heads.append(d)
        self.heads = nn.ModuleList(self.heads)
        self.w_o = nn.Linear(self.num_heads * self.d_v, self.hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, mask):
        attentions = []
        for i in range(self.num_heads):
            d = self.heads[i]
            w_q, w_k, w_v = d['w_q'], d['w_k'], d['w_v']

            q = w_q(input.permute(0, 2, 1))  # (batch_size, seq_len, d_q)
            k = w_k(input.permute(0, 2, 1))  # (batch_size, seq_len, d_k)
            v = w_v(input.permute(0, 2, 1))  # (batch_size, seq_len, d_v)
            qk = torch.bmm(q, k.transpose(1, 2))  # (batch_size, seq_len, seq_len)
            qk = torch.mul(qk, 1 / math.sqrt(self.d_k))
            qk = qk.masked_fill(1 - mask.unsqueeze(1), -np.inf)
            qk = self.softmax(qk)
            qkv = torch.bmm(qk, v)  # (batch_size, l_s, d_v)
            attentions.append(qkv)
            del q, k, v, qk

        combined_head = torch.cat(attentions, dim=2)  # (batch_size, seq_len, num_heads * d_v)
        attention = self.w_o(combined_head).permute(0, 2, 1)
        return attention

# adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html
class PostionalEncoder(nn.Module):
    def __init__(self, hidden_size, max_len=1000):
        super(PostionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, hidden_size, 2) *
                             -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(2)].permute(0, 2, 1)


class EncoderBlock(nn.Module):
    def __init__(self, L, bid, enc_convs, hidden_size, enc_heads, conv_kernal_size, conv_padding, drop_prob):
        super(EncoderBlock, self).__init__()

        self.bid = bid # id of the block inside a stack (goes from 0 to stack size -1)
        self.L = L # Total number of layers "module" (question / context encoder [6] or model encoder [21])
        self.enc_convs = enc_convs
        self.drop_prob = drop_prob

        self.pos_enc = PostionalEncoder(hidden_size)
        self.convolutions = nn.ModuleList([ConvUnit_DS(hidden_size, hidden_size, conv_kernal_size, conv_padding) for x in range(enc_convs)])
        self.attn = SelfAttention(hidden_size, enc_heads, 96, 96)

        self.layer_norm_conv = nn.ModuleList([nn.LayerNorm(hidden_size) for x in range(enc_convs)])
        self.layer_norm_self_att = nn.LayerNorm(hidden_size)
        self.layer_norm_ff = nn.LayerNorm(hidden_size)

        self.ff_1 = ConvUnit(hidden_size, hidden_size, kernel_size=1, padding=0, relu=True, bias=True)
        self.ff_2 = ConvUnit(hidden_size, hidden_size, kernel_size=1, padding=0, relu=False, bias=True)

    def forward(self, input, mask):
        x = self.pos_enc(input)
        l = self.bid * (self.enc_convs + 2)

        for i in range(self.enc_convs):
            residual = x
            l += 1
            x = self.layer_norm_conv[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            if (i) % 2 == 0:  # dropout rate between every two layers is 0.1
                x = F.dropout(x, p=self.drop_prob, training=self.training)
            x = self.convolutions[i](x)
            x = self.stochastic_dropout(x, residual, self.drop_prob, l, self.L)

        # Paper says, We also adopt the stochastic depth method (layer dropout) (Huang et al., 2016)
        # within each embedding or model encoder layer
        # We guess, the following are also considered as "layers"

        # Self-Attention layer
        residual = x
        l += 1
        x = self.layer_norm_self_att(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.attn(x, mask)
        x = self.stochastic_dropout(x, residual, self.drop_prob, l, self.L)

        # FeedForward layer
        # Per the transformer paper (https://arxiv.org/pdf/1706.03762.pdf), in addition to attention sub-layers,
        # each of the layers in our encoder and decoder contains a fully
        # connected feed-forward network, which is applied to each position separately and identically. This
        # consists of two linear transformations with a ReLU activation in between
        # We use ConvUnit_ID because it is perhaps faster than the linear layer. The ReLU is handled inside the conv.
        residual = x
        l += 1
        x = self.layer_norm_ff(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.dropout(x, p=self.drop_prob, training=self.training)  # This violates the every every two layers principle, not sure.
        x = self.ff_1(x)
        x = self.ff_2(x)
        x = self.stochastic_dropout(x, residual, self.drop_prob, l, self.L)
        return x


    def stochastic_dropout(self, curr, prev, pL, l, L):
        # print(self.bid, l, L, pL)
        if self.training:
            bl = torch.empty(1).uniform_(0, 1)
            pl = pL * float(l) / L
            if bl < pl:
                return prev
            else:
                return F.dropout(curr, pl, training=self.training) + prev
        else:
            return curr + prev


class StackedEncodeBlock(nn.Module):
    def __init__(self, L, enc_m_blocks, enc_m_convs, hidden_size, enc_heads, conv_kernal_size, conv_padding, drop_prob):
        super(StackedEncodeBlock, self).__init__()
        self.num_blocks = enc_m_blocks
        self.enc_m_convs = enc_m_convs

        self.blocks = nn.ModuleList(
            [EncoderBlock(L, bid, enc_m_convs, hidden_size, enc_heads, conv_kernal_size, conv_padding, drop_prob) for bid
             in range(enc_m_blocks)])

    def forward(self, input, mask):
        x = input
        for i in range(self.num_blocks):
            x = self.blocks[i](x, mask)
        return x

class Embedding(nn.Module):
    def __init__(self, e_char, e_word, hidden_size, drop_prob_char, drop_prob):
        super().__init__()
        self.drop_prob_char = drop_prob_char
        self.drop_prob = drop_prob
        self.high = HighwayUnit(2, hidden_size, drop_prob)
        self.conv_1d = ConvUnit(e_word + hidden_size, hidden_size, 1, 0, relu = False, bias=False)
        self.conv_2d = nn.Conv2d(e_char, hidden_size, kernel_size = (1, 7), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv_2d.weight, nonlinearity='relu')

    def forward(self, emb_c, emb_w):
        # in the paper they emit 500 dimensions from embedding and then convolve down to hidden_size.
        # Here we directly convolve down to hidden_size thanks to 2D convolution
        emb_c = emb_c.permute(0, 3, 1, 2) # (batch_size, seq_len, m_word, e_char)
        emb_c = F.dropout(emb_c, p=self.drop_prob_char, training=self.training)
        emb_c = self.conv_2d(emb_c) # (batch_size, hidden_size, seq_len, e_char - kernel_size[1])
        emb_c = F.relu(emb_c)
        emb_c, _ = torch.max(emb_c, dim=3) # (batch_size, hidden_size, seq_len)

        emb_w = F.dropout(emb_w, p=self.drop_prob, training=self.training)
        emb_w = emb_w.transpose(1, 2)

        combined = torch.cat([emb_c, emb_w], dim=1) # (batch_size, hidden_size + e_word, seq_len)
        combined = self.conv_1d(combined)
        combined = self.high(combined) # (batch_size, hidden_size, seq_len)
        return combined

class QANetOutput(nn.Module):
    def __init__(self, hidden_size):
        super(QANetOutput, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, 1)

    def forward(self, sb_a, sb_b, mask):
        # sb1, sb2 - (batch_size, hidden_size, c_len)
        stacks = torch.cat((sb_a, sb_b), 1).permute(0, 2, 1)  # (batch_size, c_len,  hidden_size * 2)
        logits = self.linear(stacks) # (batch_size, c_len,1)
        logits = logits.squeeze(dim=2) # (batch_size, c_len)
        prob = masked_softmax(logits, mask, log_softmax=True) # (batch_size, c_len)
        return prob

class HighwayUnit(nn.Module):
    def __init__(self, num_layers, hidden_size, drop_prob):
        super().__init__()
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.transform = nn.ModuleList([ConvUnit(hidden_size, hidden_size, 1, 0, relu=False, bias=True) for _ in range(num_layers)])
        self.gate = nn.ModuleList([ConvUnit(hidden_size, hidden_size, 1, 0, relu=False, bias=True) for _ in range(num_layers)])

    def forward(self, x):
        for l in range(self.num_layers):
            gate = torch.sigmoid(self.gate[l](x))
            transformed = self.transform[l](x)
            transformed = F.dropout(transformed, p=self.drop_prob, training=self.training)
            x = gate * transformed + (1 - gate) * x
        return x


class ConvUnit(nn.Module):
    def __init__(self, in_channels, num_filter, kernel_size, padding, bias=True, relu=True):
        super(ConvUnit, self).__init__()
        self.relu = relu
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=num_filter, kernel_size=kernel_size, bias=bias,
                             padding=padding)
        nn.init.xavier_uniform_(self.cnn.weight)

    def forward(self, input):
        x_conv = self.cnn(input)
        if self.relu == True:
            x_conv = F.relu(x_conv)
        return x_conv

class ConvUnit_DS(nn.Module):
    # refer to: https://arxiv.org/abs/1706.03059
    def __init__(self, in_channels, num_filter, kernel_size, bias=True):
        super().__init__()
        self.dconv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2, bias=False)
        self.pconv = nn.Conv1d(in_channels=in_channels, out_channels=num_filter, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pconv(self.dconv(x)))