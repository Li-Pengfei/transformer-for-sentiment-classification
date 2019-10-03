import copy
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer
from .Constants import *

from global_random_seed import RANDOM_SEED

# make everything reproducible
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def position_encoding_init(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_padding_mask(seq_q, seq_k):
    ''' For masking out the padding part. '''

    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)   # b x lk
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_output_padding_mask(seq_q, d_model):
    ''' For masking out the padding part. For output '''
    padding_mask = seq_q.eq(PAD)   # b x lq
    padding_mask = padding_mask.unsqueeze(2).expand(-1, -1, d_model)  # b x lq x d_model
    return padding_mask


class Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=3, n_head=1, d_k=360, d_v=360,
            d_word_vec=360, d_model=360, d_inner_hid=720, dropout=0.1, scaled_dropout=0.1, obj_sub_pos=False,
            use_batch_norm=True, residual_bool=False, diagonal_positional_attention=False, relative_pos_dim=50,
            relative_positions=False, temper_value=0.5
    ):

        super().__init__()

        n_position = n_max_seq + 1
        # calculate max number of distances for position embedding matrix creation
        max_distance_dpa = 2 * n_max_seq + 2
        if relative_positions:
            max_distance_sujobj = 20
        else:
            max_distance_sujobj = max_distance_dpa

        self.n_max_seq = n_max_seq
        self.d_model = d_model

        # new diagonal positional encodings
        self.diagonal_positional_attention = diagonal_positional_attention
        self.relative_positions = relative_positions

        # decide whether to add subject and object positional vectors to the normal positional vectors
        self.obj_sub_pos = obj_sub_pos

        # make sure all dimensions are correct, based on the paper
        assert d_word_vec == d_model

        if obj_sub_pos:

            # embeddings for object pos encodings
            # TODO: do we need to learn separate encodings here?
            self.position_enc2 = nn.Embedding(max_distance_sujobj, d_word_vec, padding_idx=PAD)
            self.position_enc2.weight.requires_grad = True
            # don't need the sinusoids here?
            # self.position_enc2.weight = nn.Parameter(n_position, d_word_vec, requires_grad=True)

            # embedding for subject pos encodings
            self.position_enc3 = nn.Embedding(max_distance_sujobj, d_word_vec, padding_idx=PAD)
            self.position_enc3.weight.requires_grad = True

        if diagonal_positional_attention:
            # needs a positional matrix double the size of embeddings

            # what is this?
            # self.position_dpa = nn.Parameter(torch.FloatTensor((n_position*2)-1, d_word_vec//n_head).cuda())
            # position_encoding_init((n_position*2)-1, d_word_vec//n_head)

            # self.position_dpa = nn.Embedding((n_position*2)-1, d_word_vec//n_head, padding_idx=PAD)
            self.position_enc_dpa = nn.Embedding(max_distance_dpa, relative_pos_dim, padding_idx=PAD)
            # initialize embeddings
            # self.position_enc_dpa.weight.data[1:, :].uniform_(-1.0, 1.0)  # keep padding dimension to be 0
            # make sure embeddings are trainable for dpa
            self.position_enc_dpa.weight.requires_grad = True

            # TODO: do we need to set weights implicitly?
            # position_enc_dpa_weight = position_encoding_init(max_distance_dpa,relative_pos_dim,padding_idx=PAD)
            # nn.init.kaiming_normal_(position_enc_dpa_weight)
            # self.position_enc_dpa.weight.data = position_enc_dpa_weight

            # print(self.position_dpa)
            # self.position_dpa2 = PositionalEncodingLookup(d_word_vec//n_head, (n_position*2)-1)
        else:
            # generate sinusoids and freeze them as embeddings
            self.position_enc = nn.Embedding.from_pretrained(
                position_encoding_init(n_position, d_word_vec, padding_idx=PAD),
                freeze=True)

        # this is for self-learned embeddings
        # in the original paper they don't use pre-trained embeddings
        # since we use glove, we skip this step
        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)

        # use deep copy based on:
        # http://nlp.seas.harvard.edu/2018/04/03/attention.html
        self.layer_stack = nn.ModuleList([
            copy.deepcopy(EncoderLayer(
                d_model=d_model,
                d_inner_hid=d_inner_hid,
                n_head=n_head,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                scaled_dropout=scaled_dropout,
                use_batch_norm=use_batch_norm,
                residual_bool=residual_bool,
                temper_value=temper_value,
                diagonal_positional_attention=diagonal_positional_attention,
                position_dpa_dim=relative_pos_dim
            ))
            for _ in range(n_layers)])

    def forward(self, enc_non_embedded, src_seq, src_pos, pe_features):

        # original use: https://github.com/jadore801120/attention-is-all-you-need-pytorch

        # Word embedding look up, already done in rnn.py
        # in the original paper they don't use pre-trained embeddings
        # since we use glove, we skip this step
        # enc_input = self.src_word_emb(src_seq)

        # TODO: try adding vectors (word vec + pos vec) instead of just appending them
        # TODO: try experimenting with character-based embeddings?

        # add positional encoding to the initial input, add emd_vec+pos_vec value by value
        # originally we used the positional vector of the sentence from 0 to n+1
        # src_seq += self.position_enc(src_pos)

        position_dpa = None

        # decide whether to add subject and object positional vectors to the normal positional vectors
        if self.obj_sub_pos:  # default no
            # add the obj/subj embeddings to the word embeddings
            # TODO: try all variants here!, also without any obj/subj encodings

            # use only the object embeddings
            # src_seq = src_seq + self.position_enc2(pe_features[1])

            # use both object and subject encodings?
            src_seq = src_seq + self.position_enc2(pe_features[1]) + self.position_enc3(pe_features[0])

        # decide whether to use relative positions or use absolute Sinusoid position encoding for self-attention
        if self.diagonal_positional_attention:

            verbose_sizes = False
            if verbose_sizes:
                print("src_seq.size():", src_seq.size())
                print("using diagonal positional encodings 0")
                print(pe_features[2])
                print(pe_features[2].size())

            # now we take the modified positional word encodings
            position_dpa = self.position_enc_dpa(pe_features[2])  # src_pos of size 2n

            if verbose_sizes:
                print("position_dpa.size():", position_dpa.size())
        else:
            src_seq += self.position_enc(src_pos)   # use absolute Sinusoid position encoding

        enc_slf_attns = list()
        enc_output = src_seq

        # add masking for attention
        enc_slf_attn_mask = get_attn_padding_mask(enc_non_embedded, enc_non_embedded)  # b x lq x lk
        # add masking for output tokens
        enc_output_mask = get_output_padding_mask(enc_non_embedded, self.d_model)  # b x lq x d_model

        # iterate over encoder layers
        for enc_layer in self.layer_stack:

            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                slf_attn_mask=enc_slf_attn_mask,
                output_mask=enc_output_mask,
                position_dpa=position_dpa,
                sentence_words=enc_non_embedded
            )

        enc_slf_attns += [enc_slf_attn]
        return enc_output, enc_slf_attns
