"""
Define common constants.
"""
TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 500

# vocab
PAD_TOKEN = '_PAD_'
PAD_ID = 0
UNK_TOKEN = '_UNK_'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

INFINITY_NUMBER = 1e12
