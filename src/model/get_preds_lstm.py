import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from model_rnn import AffixPredictor
from data_helpers_rnn import *
from evaluation import *
from segmentation_tools import *


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # Load GloVe vectors
    print('Load pretrained word embeddings...')

    glove_vecs = dict()

    with open('glove.42B.300d.txt', encoding='utf8') as f:
        for l in f:
            glove_vecs[l.strip().split()[0]] = np.array([float(i) for i in l.strip().split()[1:]])

    # Add special tokens for start and end of sentence (randomly initialized)
    glove_vecs['<s>'] = np.random.rand(300)
    glove_vecs['</s>'] = np.random.rand(300)

    w2id = {w: i + 2 for i, w in enumerate(glove_vecs.keys())}

    glove_matrix = np.zeros((len(w2id) + 2, 300))

    for w in w2id:
        glove_matrix[w2id[w]] = glove_vecs[w]

    glove_matrix[1] = glove_matrix.mean(axis=0)

    print('Done.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')

    args = parser.parse_args()

    best_models = [
        ('pfx', 1, 'shared', '1e-03'),
        ('pfx', 2, 'shared', '1e-03'),
        ('pfx', 4, 'shared', '1e-03'),
        ('pfx', 8, 'shared', '1e-03'),
        ('pfx', 16, 'shared', '3e-04'),
        ('pfx', 32, 'shared', '1e-04'),
        ('pfx', 64, 'shared', '3e-04'),

        ('pfx', 1, 'split', '3e-04'),
        ('pfx', 2, 'split', '3e-04'),
        ('pfx', 4, 'split', '3e-04'),
        ('pfx', 8, 'split', '3e-04'),
        ('pfx', 16, 'split', '1e-04'),
        ('pfx', 32, 'split', '1e-04'),
        ('pfx', 64, 'split', '3e-04'),

        ('sfx', 1, 'shared', '3e-04'),
        ('sfx', 2, 'shared', '1e-03'),
        ('sfx', 4, 'shared', '1e-03'),
        ('sfx', 8, 'shared', '1e-03'),
        ('sfx', 16, 'shared', '3e-04'),
        ('sfx', 32, 'shared', '3e-04'),
        ('sfx', 64, 'shared', '3e-04'),

        ('sfx', 1, 'split', '3e-04'),
        ('sfx', 2, 'split', '3e-04'),
        ('sfx', 4, 'split', '3e-04'),
        ('sfx', 8, 'split', '3e-04'),
        ('sfx', 16, 'split', '3e-04'),
        ('sfx', 32, 'split', '1e-04'),
        ('sfx', 64, 'split', '1e-04'),

        ('both', 1, 'shared', '3e-03'),
        ('both', 2, 'shared', '3e-03'),
        ('both', 4, 'shared', '3e-03'),
        ('both', 8, 'shared', '3e-04'),
        ('both', 16, 'shared', '3e-04'),
        ('both', 32, 'shared', '3e-04'),
        ('both', 64, 'shared', '3e-04'),

        ('both', 1, 'split', '3e-04'),
        ('both', 2, 'split', '3e-03'),
        ('both', 4, 'split', '3e-04'),
        ('both', 8, 'split', '3e-04'),
        ('both', 16, 'split', '3e-03'),
        ('both', 32, 'split', '3e-03'),
        ('both', 64, 'split', '3e-04')
    ]

    for bm in best_models:

        print('Mode: {}'.format(bm[0]))
        print('Count: {}'.format(bm[1]))
        print('Lexicon setting: {}'.format(bm[2]))
        print('Learning rate: {}'.format(bm[3]))

        print('Batch size: {}'.format(args.batch_size))

        # Define poath to data
        inpath = str(Path('../../data/final').resolve())

        if bm[2] == 'shared':
            train_path = '{}{}sents_{:02d}_train.txt'.format(inpath, os.sep, bm[1])
            test_path = '{}{}sents_{:02d}_test.txt'.format(inpath, os.sep, bm[1])

        elif bm[2] == 'split':
            train_path = '{}{}sents_{:02d}_train_split.txt'.format(inpath, os.sep, bm[1])
            test_path = '{}{}sents_{:02d}_test_split.txt'.format(inpath, os.sep, bm[1])

        # Initialize train loader
        print('Load training data...')
        try:
            train_data = AffixDataset(train_path, bm[0])
        except FileNotFoundError:
            print('Bin not found.')
            continue

        print('Done.')

        sents_collator = SentsCollator(w2id, train_data.c2id, train_data.l2id)

        # Initialize val loader
        print('Load validation data...')
        test_data = AffixDataset(test_path, bm[0])
        print('Done.')
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=sents_collator)

        # Define device
        device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        INPUT_DIM_C = len(train_data.c_vocab) + 2
        OUTPUT_DIM = len(set(train_data.labels))
        EMBEDDING_DIM_C = 100
        HIDDEN_DIM_W = 100
        HIDDEN_DIM_C = 100
        DROPOUT = 0.2

        # Initialize model
        affix_predictor = AffixPredictor(glove_matrix, INPUT_DIM_C, EMBEDDING_DIM_C, HIDDEN_DIM_W, HIDDEN_DIM_C, OUTPUT_DIM, DROPOUT)

        # Load finetuned model weights
        print('Loading finetuned model weights from model_lstm_{}_{}_{:02d}_{}.torch...'.format(bm[2], bm[0], bm[1], bm[3]))
        affix_predictor.load_state_dict(torch.load('trained_main/model_lstm_{}_{}_{:02d}_{}.torch'.format(bm[2], bm[0], bm[1], bm[3]), map_location=device))

        # Move model to CUDA
        affix_predictor = affix_predictor.to(device)

        affix_predictor.eval()

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in test_loader:

                s_1, s_2, b, l = batch

                s_1, s_2, b, l = s_1.to(device), s_2.to(device), b.to(device), l.to(device)

                output = affix_predictor(s_1, s_2, b)

                vals, preds = torch.topk(output, k=output.size(-1), dim=-1)

                y_true.extend(l.tolist())
                y_pred.extend(preds.tolist())

        with open('results_final/results_lstm_{}_{}_macro.txt'.format(bm[2], bm[0]), 'a+') as f:
            f.write('{:.3f} & '.format(np.mean(list(mrr_macro(y_true, y_pred, 10).values()))))
        with open('results_final/results_lstm_{}_{}_micro.txt'.format(bm[2], bm[0]), 'a+') as f:
            f.write('{:.3f} & '.format(mrr_micro(y_true, y_pred, 10)))


if __name__ == '__main__':
    main()
