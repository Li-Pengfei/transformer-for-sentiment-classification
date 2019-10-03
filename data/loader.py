import json
import pickle
import random
import os
import torch
import numpy as np

from tqdm import tqdm

from utils import constant, helper, vocab
from global_random_seed import RANDOM_SEED

"""
Data loader for Yelp review json files.
"""

PAD = 0
ABS_MAX_LEN = constant.MAX_LEN

# make everything reproducible
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):

        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation


        # Construct distance mapping: map distance to an integer index
        # diagonal_positional_attention distance mapping
        self.distanceMapping_dpa = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
        self.minDistance_dpa = -(ABS_MAX_LEN - 1)
        self.maxDistance_dpa = ABS_MAX_LEN - 1
        for dis in range(self.minDistance_dpa, self.maxDistance_dpa + 1):
            self.distanceMapping_dpa[dis] = len(self.distanceMapping_dpa)


        # read the json file with data
        with open(filename) as infile:
            data = json.load(infile)

        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        self.labels = [d[-1] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """

        processed = list()
        # max_sequence_length = 0 # it's 96 now

        for i, d in enumerate(tqdm(data)):

            tokens = d['token']

            # lowercase all tokens
            if opt['lower']:
                # print("LOWERIN")
                tokens = [t.lower() for t in tokens]

            tokens = map_to_ids(tokens, vocab.word2id)

            l = len(tokens)

            # create word positional vector for self-attention
            inst_position = list([pos_i + 1 if w_i != PAD else 0 for pos_i, w_i in enumerate(tokens)])
            # print("inst_position", inst_position)

            # one-hot encoding for relation classes
            relation = d['label'] - 1   # strat from 0

            # return vector of the whole partitioned data
            processed += [(tokens, inst_position, relation)]

        return processed



    @staticmethod
    def bin_positions(positions_list):
        """
        Recalculate the word positions by binning them:
        e.g. input = [-3 -2 -1  0  1  2  3  4  5  6  7]
              --> output=[-2 -2 -1  0  1  2  2  3  3  3  3]

        :param positions_list: list of word positions relative to the query or object
        :return: new positions
        """
        # a = np.array(positions_list)
        # a[a > 0] = np.floor(np.log2(a[a > 0])) + 1
        # a[a < 0] = -np.floor(np.log2(-a[a < 0])) - 1

        """
        Recalculate the word positions by binning them:
        e.g.      input = [-3 -2 -1  0  1  2  3  4  5  6  7 8 9]
              --> output =[-3 -2 -1  0  1  2  3  3  4  4  4 4 5]
        """
        a = np.array(positions_list)
        a[a > 2] = np.ceil(np.log2(a[a > 2])) + 1
        a[a < -2] = -np.ceil(np.log2(-a[a < -2])) - 1
        return a.tolist()

    @staticmethod
    def map_distance(positions, distanceMapping, minDistance, maxDistance):
        positions_int = []
        for pos in positions:
            if pos in distanceMapping:
                positions_int.append(distanceMapping[pos])
            elif pos > maxDistance:
                positions_int.append(distanceMapping['GreaterMax'])
            else:
                positions_int.append(distanceMapping['LowerMin'])
        return positions_int

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """

        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))  # transpose batch
        assert len(batch) == 3

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # handle word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # get_long_tensor creates a matrix out of list of lists
        # convert to tensors
        # also do padding here, it will get the longest sequence and pad the rest
        words = get_long_tensor(words, batch_size)  # matrix of tokens

        # get overall relative positions for self-attention
        # double the number of positions for the diagonal positional attention
        """
        Example:
            original: 
                [1, 2, 3, 4, 5, 6, 7, 8, 9]   seq length = 9
            relativated non-binned:
                [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            relativated binned:
                [-4, -4, -4, -4, -3, -3, -2, -1, 0, 1, 2, 3, 3, 4, 4, 4, 4]
        """
        seq_len = words.size()[1]   # get seq length for the batch
        relative_positions = range(1 - seq_len, seq_len)  # length: 2*seq_len-1
        # map to integer
        relative_positions = self.map_distance(relative_positions, self.distanceMapping_dpa,
                                               self.minDistance_dpa, self.maxDistance_dpa)
        # create batch tensor
        relative_positions_dpa = torch.tensor(relative_positions).repeat(batch_size, 1)

        src_pos = get_long_tensor(batch[1], batch_size)  # matrix, positional ids for all words in sentence

        # new masks with positional padding
        masks = torch.eq(words, 0)  # should we also do +src_pos?

        labels = torch.LongTensor(batch[2])  # list of relation labels for this batch

        return (words, masks, relative_positions_dpa, src_pos, labels, orig_idx)  # notice the order

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)





def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    # print(start_idx, end_idx, length)
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))


def get_position_modified(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    # print(start_idx, end_idx, length)
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))


def get_long_tensor(tokens_list, batch_size):
    """
    Convert list of list of tokens to a padded LongTensor.
    Also perform padding here.
    """

    token_len = max(len(x) for x in tokens_list)
    token_len = min(token_len, ABS_MAX_LEN)

    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)

    for i, s in enumerate(tokens_list):
        if len(s) <= token_len:
            tokens[i, :len(s)] = torch.LongTensor(s)
        else:
            tokens[i, :] = torch.LongTensor(s[:token_len])
    return tokens


def sort_all(batch, lens):
    """
    Sort all fields by descending order of lens, and return the original indices.
    """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """
    Randomly dropout tokens (IDs) and replace them with <UNK> tokens.
    """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x for x in tokens]
