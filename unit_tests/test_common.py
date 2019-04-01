import util
from util import collate_fn, SQuAD
import torch
import torch.utils.data as data

class TestCommon(object):

    def __init__(self, args):
        # Get embeddings
        self.word_vectors = util.torch_from_json(args.word_emb_file) # (vm_len, e_word) = (88714, 300)

        # Get character embedding
        self.char_vectors = util.torch_from_json(args.char_emb_file) # (vc_len, e_char) = (1376, 64)

        self.train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
        self.train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       collate_fn=collate_fn)
        self.e_word = self.word_vectors.shape[1]
        self.args = args

        self.batch, self.m_word = 4, 7 # (IMPORTANT: m_word has to >= num_filters!!!!!)
        self.l_q, self.l_c =  10, 12
        self.v_word, self.v_char = 35, 26
        self.e_word, self.e_char = 8, 5
        self.hidden_size = 9



    def get_dummy_batch(self):
        q_len = torch.LongTensor(self.batch).random_(1, self.l_q)
        c_len = torch.LongTensor(self.batch).random_(1, self.l_c)

        qw_idxs = torch.zeros(self.batch, self.l_q).long()
        qc_idxs = torch.zeros(self.batch, self.l_q, self.m_word).long()

        cw_idxs = torch.zeros(self.batch, self.l_c).long()
        cc_idxs = torch.zeros(self.batch, self.l_c, self.m_word).long()

        for b in range(self.batch):
            qw_idxs[b, 0:q_len[b]] = torch.LongTensor(1, q_len[b]).random_(1, self.v_word)
            qc_idxs[b, 0:q_len[b], :] = torch.LongTensor(1, q_len[b], self.m_word).random_(1, self.v_char)
            cw_idxs[b, 0:c_len[b]] = torch.LongTensor(1, c_len[b]).random_(1, self.v_word)
            cc_idxs[b, 0:c_len[b], :] = torch.LongTensor(1, c_len[b], self.m_word).random_(1, self.v_char)
        return cw_idxs, cc_idxs, qw_idxs, qc_idxs

    def get_dummy_embbedding(self):
        w_emb = torch.rand(self.v_word, self.e_word)
        c_emb = torch.rand(self.v_char, self.e_char)
        return w_emb, c_emb
