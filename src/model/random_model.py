import argparse
import os
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from data_helpers import *
from evaluation import *
import random
from segmentation_tools import *
from torch.utils.data import DataLoader


def train(train_loader, val_loader, mode, lexicon):

    print('Training model...')

    labels_counter = Counter()

    for batch in train_loader:

        sents, masks, segs, idxes_mask, labels = batch

        if mode == 'pfx' or mode == 'sfx':

            labels_counter.update(labels.tolist())

        elif mode == 'both':

            labels_counter.update([tuple(l) for l in labels.tolist()])

    preds = list(labels_counter.keys())

    mrr_micro, mrr_macro_dict = test(val_loader, mode, preds)

    mrr_macro = np.mean(list(mrr_macro_dict.values()))

    print('MRR@10 (micro):\t{:.4f}'.format(mrr_micro))
    print('MRR@10 (mcro):\t{:.4f}'.format(mrr_macro))
    print('Best:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    print('Worst:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1])[:5])

    with open('results_final/results_random_{}_{}_micro.txt'.format(lexicon, mode), 'a+') as f:
        f.write('{:.3f} & '.format(mrr_micro))

    with open('results_final/results_random_{}_{}_macro.txt'.format(lexicon, mode), 'a+') as f:
        f.write('{:.3f} & '.format(mrr_macro))


def test(test_loader, mode, preds):

    print('Evaluating model...')

    y_true = list()
    y_pred = list()

    for batch in test_loader:

        sents, masks, segs, idxes_mask, labels = batch

        # Store labels and predictions
        if mode == 'pfx' or mode == 'sfx':
            y_true.extend(labels.tolist())
        elif mode == 'both':
            y_true.extend([tuple(l) for l in labels.tolist()])

        for _ in range(len(labels)):
            y_pred.append(random.sample(preds, len(preds)))

    return mrr_micro(y_true, y_pred, 10), mrr_macro(y_true, y_pred, 10)


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default=None, type=str, required=True, help='Affix type.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting')

    args = parser.parse_args()

    for count in [1, 2, 4, 8, 16, 32, 64]:

        print('Mode: {}'.format(args.mode))
        print('Count: {}'.format(count))
        print('Lexicon setting: {}'.format(args.lexicon))
        print('Batch size: {}'.format(args.batch_size))

        # Define poath to data
        inpath = str(Path('../../data/final').resolve())

        if args.lexicon == 'shared':
            train_path = '{}{}sents_{:02d}_train.txt'.format(inpath, os.sep, count)
            test_path = '{}{}sents_{:02d}_test.txt'.format(inpath, os.sep, count)

        elif args.lexicon == 'split':
            train_path = '{}{}sents_{:02d}_train_split.txt'.format(inpath, os.sep, count)
            test_path = '{}{}sents_{:02d}_test_split.txt'.format(inpath, os.sep, count)

        # Initialize train loader
        print('Load training data...')
        try:
            train_data = AffixDataset(train_path, args.mode)
        except FileNotFoundError:
            print('Bin not found.')
            continue

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sents)

        # Initialize val loader
        print('Load validation data...')
        test_data = AffixDataset(test_path, args.mode)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_sents)

        train(train_loader, test_loader, args.mode, args.lexicon)


if __name__ == '__main__':
    main()
