"""
Run evaluation with saved models.
"""

import os
import random
import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from global_random_seed import RANDOM_SEED

from data.loader import DataLoader, KnowledgeLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', type=str, help='Directory of the model.',
    default="saved_models/tmp_model/"
)
# parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str,
                    default="saved_models/out/",
                    help="Save model predictions to this dir."
                    )

parser.add_argument('--seed', type=int, default=RANDOM_SEED)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--use_knowledge', action='store_true')

args = parser.parse_args()

with open('global_random_seed.py', 'w') as the_file:
    the_file.write('RANDOM_SEED = ' + str(args.seed))

# set top level random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)

if args.cpu:
    args.cuda = False
elif args.cuda:
    # set random seed for cuda as well
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
# TODO: are we using dropout in testing??
# opt["dropout"] = 0.0
# opt["scaled_dropout"] = 0.0

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load knowledge indicators
if args.use_knowledge:
    RI_tuple_file = 'dataset/vocab/RI_tuple_list.pkl'
    # RI_cluster_file='dataset/vocab/RI_cluster_index.pkl'
    # knowledge_loader = KnowledgeLoader(vocab, RI_tuple_file, RI_cluster_file)
    knowledge_loader = KnowledgeLoader(vocab, RI_tuple_file)
else:
    knowledge_loader = None

model = RelationModel(opt, knowledge_indicator=knowledge_loader)
model.load(model_file)


# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])

predictions = []
# all_probs = []
data_visual = []

with torch.no_grad():
    for i, b in enumerate(batch):

        batch_data = batch.data[i]
        batch_size = len(batch_data)
        batch_data = list(zip(*batch_data))
        assert len(batch_data) == 10
        tokens = batch_data[0]
        labels = batch_data[9]

        preds, probs, loss, weights = model.predict(b)

        for j in range(batch_size):
            text = [vocab.id2word[idx] for idx in tokens[j]]
            label = labels[j]
            prediction = preds[j]
            posterior = probs[j]
            attention = weights[j].tolist()
            jason_dict = {
                'text': text,
                'label': label,
                'prediction': prediction,
                'posterior': posterior,
                'attention': attention,
                'id': 'sample{}_{}'.format(i, j)
            }
            data_visual.append(jason_dict)

        predictions += preds
        # all_probs += probs

# print(data_visual[0:4])

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    # with open(args.out + 'test_temp.pkl', 'wb') as outfile:
    #     pickle.dump(all_probs, outfile)
    # print("Prediction scores saved to {}.".format(args.out))

    with open(args.out + 'predictions.pkl', 'wb') as outfile:
        pickle.dump(predictions, outfile)
    print("Prediction saved to {}.".format(args.out))
    with open(args.out + 'gold.pkl', 'wb') as outfile:
        pickle.dump(batch.gold(), outfile)
    print("True label saved to {}.".format(args.out))

    with open(args.out + 'data_visual.json', 'w') as outfile:
        json.dump(data_visual, outfile)
    print("Json file for data visualization saved to {}.".format(args.out))

print("Evaluation ended.")
