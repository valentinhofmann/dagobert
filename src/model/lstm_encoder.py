import warnings

warnings.filterwarnings('ignore')

import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_helpers_rnn import *
from model_rnn import AffixPredictor
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

    parser.add_argument('--count', default=None, type=int, required=True, help='Count of derivatives.')
    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting')
    parser.add_argument('--mode', default=None, type=str, required=True, help='Affix type.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')

    args = parser.parse_args()

    print('Mode: {}'.format(args.mode))
    print('Lexicon setting: {}'.format(args.lexicon))
    print('Count: {}'.format(args.count))
    print('Batch size: {}'.format(args.batch_size))
    print('Learning rate: {}'.format(args.lr))
    print('Number of epochs: {}'.format(args.n_epochs))

    # Define poath to data
    inpath = str(Path('../../data/final').resolve())

    if args.lexicon == 'shared':
        train_path = '{}{}sents_{:02d}_train.txt'.format(inpath, os.sep, args.count)
        val_path = '{}{}sents_{:02d}_dev.txt'.format(inpath, os.sep, args.count)

    elif args.lexicon == 'split':
        train_path = '{}{}sents_{:02d}_train_split.txt'.format(inpath, os.sep, args.count)
        val_path = '{}{}sents_{:02d}_dev_split.txt'.format(inpath, os.sep, args.count)

    # Initialize train loader
    print('Load training data...')
    train_data = AffixDataset(train_path, args.mode)

    print('Done.')

    sents_collator = SentsCollator(w2id, train_data.c2id, train_data.l2id)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=sents_collator)

    # Initialize val loader
    print('Load validation data...')
    val_data = AffixDataset(val_path, args.mode)
    print('Done.')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=sents_collator)

    INPUT_DIM_C = len(train_data.c_vocab) + 2
    OUTPUT_DIM = len(set(train_data.labels))
    EMBEDDING_DIM_C = 100
    HIDDEN_DIM_W = 100
    HIDDEN_DIM_C = 100
    DROPOUT = 0.2
    N_EPOCHS = args.n_epochs

    affix_predictor = AffixPredictor(glove_matrix, INPUT_DIM_C, EMBEDDING_DIM_C, HIDDEN_DIM_W, HIDDEN_DIM_C, OUTPUT_DIM, DROPOUT)

    # Define device
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    # Define training objective
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(affix_predictor.parameters(), lr=args.lr)

    # Move model to CUDA
    affix_predictor = affix_predictor.to(device)

    best_mrr = None

    # Train model
    print('Training model...')
    for epoch in range(N_EPOCHS):

        for i, batch in enumerate(train_loader):

            if i % 1000 == 0:
                print(i * args.batch_size / len(train_data))

            s_1, s_2, b, l = batch

            if s_1.size(-1) == 0 or s_2.size(-1) == 0:
                continue

            s_1, s_2, b, l = s_1.to(device), s_2.to(device), b.to(device), l.to(device)

            affix_predictor.train()

            optimizer.zero_grad()

            output = affix_predictor(s_1, s_2, b)

            loss = criterion(output, l)

            loss.backward()

            optimizer.step()

        affix_predictor.eval()

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in val_loader:

                s_1, s_2, b, l = batch

                s_1, s_2, b, l = s_1.to(device), s_2.to(device), b.to(device), l.to(device)

                output = affix_predictor(s_1, s_2, b)

                vals, preds = torch.topk(output, k=output.size(-1), dim=-1)

                y_true.extend(l.tolist())
                y_pred.extend(preds.tolist())

        print('Epoch {}'.format(epoch))
        print('Micro:', mrr_micro(y_true, y_pred, 10))
        print('Macro:', np.mean(list(mrr_macro(y_true, y_pred, 10).values())))

        with open('results_main/results_lstm_{}_{}_{:02d}_micro.txt'.format(args.lexicon, args.mode, args.count), 'a+') as f:
            f.write('{:.0e}\t{}\n'.format(args.lr, mrr_micro(y_true, y_pred, 10)))
        with open('results_main/results_lstm_{}_{}_{:02d}_macro.txt'.format(args.lexicon, args.mode, args.count), 'a+') as f:
            f.write('{:.0e}\t{}\n'.format(args.lr, np.mean(list(mrr_macro(y_true, y_pred, 10).values()))))

        if best_mrr is None or np.mean(list(mrr_macro(y_true, y_pred, 10).values())) > best_mrr:

            best_mrr = np.mean(list(mrr_macro(y_true, y_pred, 10).values()))

            torch.save(affix_predictor.state_dict(), 'trained_main/model_lstm_{}_{}_{:02d}_{:.0e}.torch'.format(args.lexicon, args.mode, args.count, args.lr))

    with open('results_main/results_lstm_{}_{}_{:02d}_micro.txt'.format(args.lexicon, args.mode, args.count), 'a+') as f:
        f.write('\n')
    with open('results_main/results_lstm_{}_{}_{:02d}_macro.txt'.format(args.lexicon, args.mode, args.count), 'a+') as f:
        f.write('\n')

if __name__ == '__main__':
    main()
