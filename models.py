"""Top-level model classes.


"""

import layers
import qanet_layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):

    def __init__(self, w_emb, c_emb, hidden_size, drop_prob=0.1):
        super(QANet, self).__init__()

        # Encoder Config
        conv_kernal_size = 7
        conv_padding = 5
        enc_heads = 1
        enc_qc_convs = 4
        enc_m_convs = 2
        enc_m_blocks = 7
        L1 = enc_qc_convs + 2
        L2 = (enc_m_convs + 2) * enc_m_blocks

        e_char = c_emb.shape[1]  # embedding size of the character
        e_word = w_emb.shape[1]  # embedding size of the word

        self.drop_prob = drop_prob

        self.emb = qanet_layers.Embedding(e_char, e_word, hidden_size, 0.05, drop_prob)
        self.word_vectors = nn.Embedding.from_pretrained(w_emb, freeze=True)
        self.char_vectors = nn.Embedding.from_pretrained(c_emb, freeze=False)

        self.cqa = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)
        self.mapper = qanet_layers.ConvUnit(hidden_size * 4, hidden_size, 1, 0, bias=False, relu=False)
        self.enc_qc = qanet_layers.EncoderBlock(L1, 0, enc_qc_convs, hidden_size, enc_heads, conv_kernal_size,
                                                conv_padding, drop_prob)

        self.enc_m = qanet_layers.StackedEncodeBlock(L2, enc_m_blocks, enc_m_convs, hidden_size, enc_heads,
                                                     conv_kernal_size, conv_padding, drop_prob)
        self.start = qanet_layers.QANetOutput(hidden_size)
        self.end = qanet_layers.QANetOutput(hidden_size)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs  # (batch_size, c_len)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs  # (batch_size, q_len)

        # Layer 1: embedding layer
        c_word = self.word_vectors(cw_idxs)
        c_char = self.char_vectors(cc_idxs)
        c = self.emb(c_char, c_word)

        q_word = self.word_vectors(qw_idxs)
        q_char = self.char_vectors(qc_idxs)
        q = self.emb(q_char, q_word)

        # Layer 2: encoding layer
        c = self.enc_qc(c, c_mask)
        q = self.enc_qc(q, q_mask)

        # Layer 3: context-question attention layer
        c_q_a = self.cqa(c.permute(0, 2, 1), q.permute(0, 2, 1), c_mask, q_mask)  # (batch_size, c_len, 4 * hidden_size)
        c_q_a = self.mapper(c_q_a.permute(0, 2, 1))  # (batch_size, hidden_size, c_len)
        c_q_a = F.dropout(c_q_a, p=self.drop_prob, training=self.training) # every two layers!

        # Layer 4: model encoding layer
        sb_1 = self.enc_m(c_q_a, c_mask)  # (batch_size, hidden_size, c_len)
        sb_2 = self.enc_m(sb_1, c_mask)  # (batch_size, hidden_size, c_len)
        sb_2 = F.dropout(sb_2, p=self.drop_prob, training=self.training) # every two layers!
        sb_3 = self.enc_m(sb_2, c_mask)  # (batch_size, hidden_size, c_len)

        # Layer 5: output layer
        prob_start = self.start(sb_1, sb_2, c_mask)  # (batch_size, c_len)
        prob_end = self.end(sb_1, sb_3, c_mask)  # (batch_size, c_len)

        return prob_start, prob_end
