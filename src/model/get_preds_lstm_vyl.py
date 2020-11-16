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
from data_helpers_rnn_vyl import *
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

    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting')
    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=str, required=True, help='Learning rate.')

    args = parser.parse_args()

    print('Lexicon setting: {}'.format(args.lexicon))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))

    # Define poath to data
    inpath = str(Path('../../data/final').resolve())

    if args.lexicon == 'shared':
        train_path = '{}{}sents_vyl_train.txt'.format(inpath, os.sep)
        test_path = '{}{}sents_vyl_test.txt'.format(inpath, os.sep)

    elif args.lexicon == 'split':
        train_path = '{}{}sents_vyl_train_split.txt'.format(inpath, os.sep)
        test_path = '{}{}sents_vyl_test_split.txt'.format(inpath, os.sep)

    # Initialize train loader
    print('Load training data...')
    train_data = AffixDataset(train_path)

    print('Done.')

    sents_collator = SentsCollator(w2id, train_data.c2id, train_data.l2id)

    # Initialize val loader
    print('Load validation data...')
    test_data = AffixDataset(test_path)
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

    print('Loading finetuned model weights from model_lstm_vyl_{}_{}.torch...'.format(args.lexicon, args.lr))
    affix_predictor.load_state_dict(torch.load('trained_vyl/model_lstm_vyl_{}_{}.torch'.format(args.lexicon, args.lr), map_location=device))

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

    acc = len([1 for t, p in zip(y_true, y_pred) if t == p[0]]) / len(y_true)

    with open('results_final/results_vyl_lstm.txt', 'a+') as f:
        f.write('{:.3f} & '.format(acc))


if __name__ == '__main__':
    main()
