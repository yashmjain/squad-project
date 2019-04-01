# -*- coding: utf-8 -*-

from args import get_train_args
import os
from unit_tests.test_common import TestCommon
from models import QANet
import qanet_layers
import layers
import torch

def test_embedding_with_dummy(tc : TestCommon):
    num_filter = 200
    word_vectors, char_vectors = tc.get_dummy_embbedding()
    embedding = qanet_layers.Embedding(word_vectors, char_vectors, num_filter)
    cw_idxs, cc_idxs, qw_idxs, qc_idxs = tc.get_dummy_batch()

    batch, l_c, m_word = cc_idxs.size()

    emb_context = embedding.forward(cw_idxs, cc_idxs)
    assert emb_context.shape == (batch, l_c, embedding.e_word + num_filter)

    emb_question = embedding.forward(qw_idxs, qc_idxs)
    batch, l_q, m_word = qc_idxs.size()
    assert emb_question.shape ==(batch, l_q, embedding.e_word + num_filter)

    print("####Test Passed: test_embedding_with_dummy ####!")
    return emb_context, emb_question, cw_idxs, qw_idxs

def test_embedding(tc : TestCommon):
    num_filter = 200
    embedding = qanet_layers.Embedding(tc.word_vectors, tc.char_vectors, num_filter)

    for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in tc.train_loader:
        # From the paper:
        #  The typical length of the paragraphs is around 250 while the question is of 10 tokens although there are exceptionally long cases.
        #  The maximum context length is set to 400 and any paragraph longer than that would be discarded. This is in words
        #  The maximum answer length is set to 30. TODO: Is it in chars? Do we need to pre-process the answers as well?

        # cw_idxs (batch_size, seq_len_c) = (64,seq_len_c)
        # qw_idxs (batch_size, seq_len_q) = (64,28)

        # cc_idxs (batch_size, max_context_len_in_words, max_word_len_in_chars_padded) = (64,seq_len_c,16)
        # qc_idxs (batch_size, max_question_len_in_words, max_word_len_in_chars_padded) = (64,seq_len_q,16)


        # Get the embedding of the context
        batch, seq_len_c, m_word = cc_idxs.size()
        emb_context = embedding.forward(cw_idxs, cc_idxs)
        assert emb_context.shape == (batch, seq_len_c, embedding.e_word + num_filter)

        # Get the embedding of the query
        emb_question = embedding.forward(qw_idxs, qc_idxs)
        batch, seq_len_q, m_word = qc_idxs.size()
        assert emb_question.shape ==(batch, seq_len_q, embedding.e_word + num_filter)
        print("####Test Passed: test_embedding ####!")
        break
    return emb_context, emb_question, cw_idxs, qw_idxs

def test_question_context_attention(tc: TestCommon):
    emb_context, emb_question, cw_idxs, qw_idxs  = test_embedding(tc)
    c_mask = torch.zeros_like(cw_idxs) != cw_idxs
    q_mask = torch.zeros_like(qw_idxs) != qw_idxs

    cqa = layers.BiDAFAttention(500, tc.args.drop_prob) # TODO: This should really be not 500 but hidden_size * 3
    cont_que_attn = cqa(emb_context, emb_question, c_mask, q_mask)
    print("####Test Passed: test_question_context_attention ####!")
         
def test_encoder(tc: TestCommon):

    NUM_FILTERS = 128
    KERNEL_SIZE = 7
    CONV_LAYER_COUNT = 4
    PADDING = 3
    DROPOUT_PROB = 0.5
    NUM_HEADS = 8



    c_emb, q_emb, cw_idxs, qw_idxs  = test_embedding(tc)
    c_mask = torch.zeros_like(cw_idxs) != cw_idxs  # (batch_size, c_len)
    q_mask = torch.zeros_like(qw_idxs) != qw_idxs  # (batch_size, q_len)

    batch_q, l_q, embedding_q = q_emb.size()
    batch_c, l_c, embedding_c = c_emb.size()
    
    encoderblock = qanet_layers.EncoderBlock(0, CONV_LAYER_COUNT,NUM_FILTERS, NUM_HEADS, KERNEL_SIZE, PADDING, DROPOUT_PROB)
    mapper1 = qanet_layers.ConvUnit(embedding_q, NUM_FILTERS, KERNEL_SIZE, PADDING)

    c_emb_mapped = mapper1(c_emb.permute(0,2,1))
    q_emb_mapped = mapper1(q_emb.permute(0,2,1))

    q_enc = encoderblock(q_emb_mapped, q_mask, 1)
    c_enc = encoderblock(c_emb_mapped, c_mask, 1)

    assert q_enc.shape == (batch_q, NUM_FILTERS, l_q)
    assert c_enc.shape == (batch_c, NUM_FILTERS, l_c)
    
    print("####Test Passed: test_encoder ####!")


def test_encoder_with_dummy(tc: TestCommon):
    NUM_FILTERS = 128
    KERNEL_SIZE = 7
    CONV_LAYER_COUNT = 4
    PADDING = 3
    DROPOUT_PROB = 0.5
    NUM_HEADS = 8

    word_vectors, char_vectors = tc.get_dummy_embbedding()
    embedding = qanet_layers.Embedding(word_vectors, char_vectors, NUM_FILTERS)
    cw_idxs, cc_idxs, qw_idxs, qc_idxs = tc.get_dummy_batch()
    c_mask = torch.zeros_like(cw_idxs) != cw_idxs  # (batch_size, c_len)
    q_mask = torch.zeros_like(qw_idxs) != qw_idxs  # (batch_size, q_len)

    c_emb = embedding.forward(cw_idxs, cc_idxs)
    q_emb = embedding.forward(qw_idxs, qc_idxs)

    batch_q, l_q, embedding_q = q_emb.size()
    batch_c, l_c, embedding_c = c_emb.size()

    encoderblock = qanet_layers.EncoderBlock(0, CONV_LAYER_COUNT, NUM_FILTERS, NUM_HEADS, KERNEL_SIZE, PADDING,DROPOUT_PROB)
    mapper1 = qanet_layers.ConvUnit(embedding_q, NUM_FILTERS, KERNEL_SIZE, PADDING)

    c_emb_mapped = mapper1(c_emb.permute(0, 2, 1))
    q_emb_mapped = mapper1(q_emb.permute(0, 2, 1))

    c_enc = encoderblock(c_emb_mapped, c_mask, 1)
    q_enc = encoderblock(q_emb_mapped, q_mask, 1)

    assert q_enc.shape == (batch_q, NUM_FILTERS, l_q)
    assert c_enc.shape == (batch_c, NUM_FILTERS, l_c)

    print("####Test Passed: test_encoder_with_dummy ####!")


def test_model(tc: TestCommon):
    qanet = QANet(tc.word_vectors, tc.char_vectors, 96, 0.1)
    #cw_idxs : context word (The number of words in the context (batch_shape * max_num_of_words)
    #cc_idxs : context char (The number of character in the context (batch_shape * max_num_of_words * max_number of character)
    #qw_idxs : query word (The number of words in the question (batch_shape * max_num_of_words)
    #qc_idxs :query char (The number of character in the question (batch_shape * max_num_of_words * max number of character)
    for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in tc.train_loader:
        _enc, q_enc = qanet(cw_idxs, cc_idxs, qw_idxs, qc_idxs)

    print("####Test Passed: test_model ####!")

def test_model_with_dummy(tc: TestCommon):
    w_emb, c_emb = tc.get_dummy_embbedding()
    cw_idxs, cc_idxs, qw_idxs, qc_idxs = tc.get_dummy_batch()
    qanet = QANet(w_emb, c_emb, tc.hidden_size)
    _enc, q_enc = qanet(cw_idxs, cc_idxs, qw_idxs, qc_idxs)

    print("####Test Passed: test_model_with_dummy ####!")

def test_positional_encoding(tc: TestCommon):
    batch_size, hidden_size, c_len = 3, 4, 7
    input = torch.arange(batch_size * hidden_size * c_len).to(torch.float).reshape(batch_size, hidden_size, c_len)
    #print(input)
    pos_enc = qanet_layers.PostionalEncoder(hidden_size)
    output = pos_enc(input)
    #print("Mode = Fixed", output)

    print("####Test Passed: test_positional_encoding ####!")


if __name__ == '__main__':
    os.environ['DATA_DIR'] = '../data-small'
    tc = TestCommon(get_train_args())
    #test_embedding(tc)
    #test_embedding_with_dummy(tc)
    #test_encoder(tc)
    #test_encoder_with_dummy(tc)
    #test_question_context_attention(tc)
    test_positional_encoding(tc)
    test_model(tc)
    

