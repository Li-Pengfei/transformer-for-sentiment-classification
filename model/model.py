from global_random_seed import RANDOM_SEED

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import constant, torch_utils
from .transformer.Models import Encoder


# make everything reproducible
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class SA_Model(object):
    """
    A wrapper class for the training and evaluation of models.
    """

    def __init__(self, opt, emb_matrix=None):

        self.opt = opt
        self.model = Our_Model(opt, emb_matrix)

        # # pass weights per class, each class corresponds to its index
        # weights = [opt['weight_no_rel']]
        # rel_classes_weights = [opt["weight_rest"]] * 41
        # weights.extend(rel_classes_weights)
        # print("Using weights", weights)
        # assert len(weights) == 42
        # class_weights = torch.FloatTensor(weights).to("cuda")

        self.criterion = nn.CrossEntropyLoss()  # weight=class_weights
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        # print(self.parameters)
        # print(len(self.parameters))

        if opt['cuda']:
            self.model.to("cuda")
            self.criterion.to("cuda")

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        """
        Run a step of forward and backward model update.
        """

        if self.opt['cuda']:
            inputs = [b.to("cuda") for b in batch[:4]]
            labels = batch[4].to("cuda")
        else:
            inputs = [b for b in batch[:4]]
            labels = batch[4]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()

        # parallel training using all available GPUs!
        if torch.cuda.device_count() > 1:
            # print("Train using", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model_parallel = nn.DataParallel(self.model)
            logits, _ = model_parallel(inputs)
        else:
            logits, _ = self.model(inputs)

        # logits, _ = self.model(inputs)
        # print(labels)
        loss = self.criterion(logits, labels)

        # backward step
        loss.backward()

        # do gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()

        loss_val = loss.item()

        return loss_val

    def predict(self, batch, unsort=True):
        """
        Run forward prediction. If unsort is True, recover the original order of the batch.
        """

        if self.opt['cuda']:
            inputs = [b.to("cuda") for b in batch[:4]]
            labels = batch[4].to("cuda")
        else:
            inputs = [b for b in batch[:4]]
            labels = batch[4]

        orig_idx = batch[5]

        # forward
        self.model.eval()

        # parallel training using all available GPUs!
        if torch.cuda.device_count() > 1:
            # print("Train using", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model_parallel = nn.DataParallel(self.model)
            logits, weights = model_parallel(inputs)
        else:
            logits, weights = self.model(inputs)

        # logits, weights = self.model(inputs)
        loss = self.criterion(logits, labels)

        probs = F.softmax(logits, dim=-1).data.cpu().numpy().tolist()

        predictions = np.argmax(probs, axis=1).tolist()
        if unsort:
            _, predictions, probs, weights = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs, weights)))]

        return predictions, probs, loss.item(), weights

    def update_lr(self, new_lr):
        """
        Update learning rate of the optimizer
        :param new_lr: new learning rate
        """
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        """
        Save the model to a file
        :param filename:
        :param epoch:
        :return:
        """
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the global representation, and f is position features,
    """

    def __init__(self, input_size, global_rep_dim, feature_size, attn_size, opt):
        super(PositionAwareAttention, self).__init__()

        self.opt = opt
        self.input_size = input_size
        self.global_rep_dim = global_rep_dim
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)

        if global_rep_dim > 0:
            self.vlinear = nn.Linear(global_rep_dim, attn_size)
        else:
            self.vlinear = None

        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size)
        else:
            self.wlinear = None

        self.tlinear = nn.Linear(attn_size, 1, bias=False)
        self.init_weights()

    def init_weights(self):

        # TODO: experiment with he and xavier
        # done, not really helping in any way here
        self.ulinear.weight.data.normal_(std=0.001).to("cuda")
        if self.vlinear is not None:
            self.vlinear.weight.data.normal_(std=0.001).to("cuda")
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001).to("cuda")
        self.tlinear.weight.data.zero_().to("cuda")  # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, f, q, lstm_units=None, lstm_layer=False):
        """
        x : input sequence , batch_size * seq_len * input_size
        q : global representaiton, batch_size * CNN_dim
        f : position embeddings, batch_size * seq_len * feature_size
        """

        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)

        if self.vlinear is not None:
            # TODO: vlinear vs ulinear, u works better, but does it make sense to share those weights?
            q_proj = self.vlinear(q.view(-1, self.global_rep_dim)).contiguous().view(
                batch_size, self.attn_size).unsqueeze(1).expand(
                batch_size, seq_len, self.attn_size)
            projs = [x_proj, q_proj]
        else:
            projs = [x_proj]

        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)
            projs.append(f_proj)

        # view in PyTorch is like reshape in numpy, view(n_rows, n_columns)
        # view(-1, n_columns) - here we define the number of columns, but n_rows will be chosen by PyTorch
        if len(projs) > 1:
            scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(batch_size, seq_len)
        else:
            scores = self.tlinear(torch.tanh(x_proj).view(-1, self.attn_size)).view(batch_size, seq_len)

        # mask padding
        # print(x_mask, x_mask.size())
        # print(x_mask.data)

        # fill elements of self tensor with value where mask is one
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=-1)

        # weighted average input vectors

        # to calculate final sentence representation z,
        # we first test two variants:

        # 1. use self-attention to calculate a_i and use lstm to get h_i
        if lstm_layer:
            outputs = weights.unsqueeze(1).bmm(lstm_units).squeeze(1)
        # 2. use self-attention for a_i and also for h_i
        else:
            outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        # add activation function
        # outputs = torch.tanh(outputs)
        # add layer norm
        # outputs = nn.functional.layer_norm(outputs, (batch_size,self.input_size))
        return outputs, weights




class Our_Model(nn.Module):
    """
    A sequence model for relation extraction.
    """

    def __init__(self, opt, emb_matrix=None):
        super(Our_Model, self).__init__()

        self.drop = nn.Dropout(opt['dropout'])
        self.drop_rnn = nn.Dropout(opt['lstm_dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)


        # add all embedding sizes to have the final input size
        # input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['entity_marker_dim']
        input_size = opt['emb_dim']

        print('Self-attn input dim:', input_size)
        print("Number of self-attn heads: ", opt["n_head"])
        print("d_v and d_k: ", input_size / opt["n_head"])

        # make sure the head units add up to n_model in integers
        assert (int(input_size / opt["n_head"])) * opt["n_head"] == input_size

        # calculate max number of distances for position embedding matrix creation
        self.max_distance_inst = constant.MAX_LEN + 1
        self.max_distance_dpa = 2 * constant.MAX_LEN + 2
        if opt['relative_positions']:
            self.max_distance_sujobj = 20
        else:
            self.max_distance_sujobj = self.max_distance_dpa

        self.self_attention_encoder = Encoder(
            n_src_vocab=55950,                   # vocab size
            n_max_seq=constant.MAX_LEN,          # max sequence length in the dataset: 96
            n_layers=opt["num_layers_encoder"],  # multiple attention+ffn layers
            d_word_vec=input_size,               # TODO: increase dim by pos and ner
            d_model=input_size,                  # d_model has to equal embedding size
            d_inner_hid=opt["hidden_self"],      # original paper: n_model * 2
            n_head=opt["n_head"],                # number of heads                    # 8
            d_k=int(input_size / opt["n_head"]),   # this should be d_model / n_heads   # 40
            d_v=int(input_size / opt["n_head"]),   # this should be d_model / n_heads   # 40
            dropout=opt['dropout'],
            scaled_dropout=opt['scaled_dropout'],
            obj_sub_pos=opt['obj_sub_pos'],      # list of obj/subj positional encodings
            use_batch_norm=opt['use_batch_norm'],
            residual_bool=opt["new_residual"],
            diagonal_positional_attention=opt["diagonal_positional_attention"],
            relative_pos_dim=opt['relative_pos_dim'],
            relative_positions=opt["relative_positions"],
            temper_value=opt["temper_value"]
        )



        # initial implementation with LSTM
        # self.rnn = nn.LSTM(
        #     input_size,
        #     opt['hidden_dim'],
        #     opt['num_layers'],  # original 2
        #     batch_first=True,
        #     dropout=opt['lstm_dropout']
        # )

        self.dense1 = nn.Linear(input_size, opt['dense_dim'])
        self.activation = nn.ReLU()   # TODO: try other activation functions
        # self.activation = nn.Tanh()
        self.linear1 = nn.Linear(opt['dense_dim'], opt['num_class'])

        if opt['attn']:
            # use positional attention from stanford paper
            self.attn_layer = PositionAwareAttention(
                input_size=input_size,  # 360, hidden_dim originally, but for self-attention should be input_size?
                # global_rep_dim=0,   # equal to input_size for 1 layer TransCNN, avg filters
                global_rep_dim=0,   # equal to input_size*kernal_size if concate filters
                feature_size=opt['pe_dim'],
                # feature_size=0,  # no position embeddings information
                attn_size=opt['attn_dim'],   # 200, attn_dim    # doesnt have to equal input size
                opt=opt
            )


            # position embedding for subj and obj (for relation extraction)
            # self.pe_emb = nn.Embedding(self.max_distance_sujobj, opt['pe_dim'], padding_idx=constant.PAD_ID)
            # position embedding for all tokens (start from 1)
            self.pe_emb = nn.Embedding(self.max_distance_inst, opt['pe_dim'], padding_idx=constant.PAD_ID)


        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):

        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)  # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)


        # initialize linear layer
        self.dense1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dense1.weight, gain=1)

        self.linear1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1)

        if self.opt['attn']:
            self.pe_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

        # decide fine-tuning
        if self.topn <= 0:
            print("Do not fine-tune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Fine-tune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Fine-tune all embeddings.")

    def zero_state(self, batch_size):
        """
        Initialize zero states for LSTM's hidden layer and cell
        """
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.to("cuda"), c0.to("cuda")
        else:
            return h0, c0

    def forward(self, inputs):

        words, masks, modified_pos_vec, inst_position = inputs  # unpack

        # seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())

        # data is split into batches in the loader.py
        # batch_size = words.size()[0]

        # embedding lookup
        word_inputs = self.emb(words)
        words = words.to("cuda")


        inputs_self = self.drop(word_inputs)     # add dropout to input
        inputs_self = inputs_self.to("cuda")
        
        ########### self-attention ###################################################################
        # should add forward here based on:
        # https://github.com/huajianjiu/attention-is-all-you-need-pytorch/commit/ed2057a741f416b5a3843fded91f84eb72414944
        outputs, enc_slf_attn = self.self_attention_encoder.forward(
            enc_non_embedded=words,
            src_seq=inputs_self, src_pos=inst_position,
            pe_features=[None, None, modified_pos_vec]
        )
        # outputs of the self-attention network
        outputs = self.drop(outputs)       # b x lq x d_model
        # ###############################################################################################

        # position-aware attention
        if self.opt['attn']:
            pe_features = self.pe_emb(inst_position)

            final_hidden, weights = self.attn_layer(outputs, masks, pe_features, None)
            # final_hidden, weights = self.attn_layer(outputs, masks, pe_features, global_rep)

        final_hidden = self.activation(self.dense1(final_hidden))
        logits = self.linear1(final_hidden)

        return logits, weights




