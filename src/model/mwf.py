import warnings

warnings.filterwarnings('ignore')

import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
from data_helpers_mwf import *
from evaluation import *
from model_mwf import *
from segmentation_tools import *
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default=None, type=str, required=True, help='Segmentation type.')
    parser.add_argument('--count', default=None, type=int, required=True, help='Count of derivatives.')
    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--freeze', default=False, action='store_true', help='Freeze BERT parameters.')

    args = parser.parse_args()

    print('Mode: {}'.format(args.mode))
    print('Lexicon setting: {}'.format(args.lexicon))
    print('Count: {}'.format(args.count))
    print('Batch size: {}'.format(args.batch_size))
    print('Learning rate: {}'.format(args.lr))
    print('Number of epochs: {}'.format(args.n_epochs))
    print('Freeze: {}'.format(args.freeze))

    # Define poath to data
    inpath = str(Path('../../data/final').resolve())

    if args.lexicon == 'shared':
        train_path = '{}{}sents_{:02d}_train.txt'.format(inpath, os.sep, args.count)
        val_path = '{}{}sents_{:02d}_dev.txt'.format(inpath, os.sep, args.count)

    elif args.lexicon == 'split':
        train_path = '{}{}sents_{:02d}_train_split.txt'.format(inpath, os.sep, args.count)
        val_path = '{}{}sents_{:02d}_dev_split.txt'.format(inpath, os.sep, args.count)

    # Load train data
    print('Load training data...')
    train_data = AffixDataset(train_path, args.mode)

    # Load val data
    print('Load validation data...')
    val_data = AffixDataset(val_path, args.mode)

    # Sample negative examples for val data
    val_data.get_full_batch()
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_sents)

    mwf_predictor = MWFPredictor(freeze=args.freeze)

    # Define device
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    # Move model to CUDA
    mwf_predictor = mwf_predictor.to(device)

    # Define training objective
    criterion = nn.BCELoss()

    # Define optimizer
    optimizer = optim.Adam(mwf_predictor.parameters(), lr=args.lr)

    # Train model
    print('Training model...')

    best_acc = None

    for epoch in range(args.n_epochs):

        # Sample negative examples for train data
        train_data.get_full_batch()
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sents)

        mwf_predictor.train()

        for i, batch in enumerate(train_loader):

            if batch[0].size(-1) > 512:
                print('Skipping batch {}...'.format(i))
                continue

            if i % 1000 == 0:
                print('Processed {} batches...'.format(i))

            sents, masks, segs, labels, idxes = batch

            sents, masks, segs, labels = sents.to(device), masks.to(device), segs.to(device), labels.to(device)

            optimizer.zero_grad()

            try:
                output = mwf_predictor(sents, masks, segs, idxes, device)
            except RuntimeError:
                print('Skipping batch {}...'.format(i))
                continue

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            del sents, masks, segs, idxes, labels, output

        print('Evaluating model...')

        mwf_predictor.eval()

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in val_loader:

                sents, masks, segs, labels, idxes = batch

                sents, masks, segs, labels = sents.to(device), masks.to(device), segs.to(device), labels.to(device)

                output = mwf_predictor(sents, masks, segs, idxes, device)

                y_true.extend(labels.tolist())

                y_pred.extend(torch.round(output).squeeze().tolist())

                del sents, masks, segs, idxes, labels, output

        acc = len([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)

        if best_acc is None or acc > best_acc:

            best_acc = acc

            if args.freeze:
                torch.save(mwf_predictor.state_dict(), 'trained_mwf/model_{}_freeze_{:02d}_{:.0e}.torch'.format(args.mode, args.count, args.lr))
            else:
                torch.save(mwf_predictor.state_dict(), 'trained_mwf/model_{}_nofreeze_{:02d}_{:.0e}.torch'.format(args.mode, args.count, args.lr))

        print('Epoch {}: {:.3f}'.format(epoch, acc))

        if args.freeze:
            with open('results_mwf/results_{}_freeze_{:02d}.txt'.format(args.mode, args.count), 'a+') as f:
                f.write('{:.0e}\t{:.3f}\n'.format(args.lr, acc))

        else:
            with open('results_mwf/results_{}_nofreeze_{:02d}.txt'.format(args.mode, args.count), 'a+') as f:
                f.write('{:.0e}\t{:.3f}\n'.format(args.lr, acc))

    if args.freeze:
        with open('results_mwf/results_{}_freeze_{:02d}.txt'.format(args.mode, args.count), 'a+') as f:
            f.write('\n')

    else:
        with open('results_mwf/results_{}_nofreeze_{:02d}.txt'.format(args.mode, args.count), 'a+') as f:
            f.write('\n')


if __name__ == '__main__':
    main()
