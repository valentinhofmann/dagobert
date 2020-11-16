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

    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')

    args = parser.parse_args()

    best_models = [
        ('bert_tok', 'freeze', 1, '1e-04'),
        ('bert_tok', 'freeze', 2, '1e-04'),
        ('bert_tok', 'freeze', 4, '1e-04'),
        ('bert_tok', 'freeze', 8, '1e-04'),
        ('bert_tok', 'freeze', 16, '1e-04'),
        ('bert_tok', 'freeze', 32, '1e-04'),
        ('bert_tok', 'freeze', 64, '1e-04'),

        ('morph_tok', 'freeze', 1, '3e-04'),
        ('morph_tok', 'freeze', 2, '3e-04'),
        ('morph_tok', 'freeze', 4, '1e-04'),
        ('morph_tok', 'freeze', 8, '1e-04'),
        ('morph_tok', 'freeze', 16, '1e-04'),
        ('morph_tok', 'freeze', 32, '1e-04'),
        ('morph_tok', 'freeze', 64, '1e-04'),

        ('bert_tok', 'nofreeze', 1, '1e-05'),
        ('bert_tok', 'nofreeze', 2, '3e-06'),
        ('bert_tok', 'nofreeze', 4, '3e-06'),
        ('bert_tok', 'nofreeze', 8, '1e-06'),
        ('bert_tok', 'nofreeze', 16, '1e-06'),
        ('bert_tok', 'nofreeze', 32, '1e-06'),
        ('bert_tok', 'nofreeze', 64, '1e-06'),

        ('morph_tok', 'nofreeze', 1, '1e-05'),
        ('morph_tok', 'nofreeze', 2, '3e-06'),
        ('morph_tok', 'nofreeze', 4, '3e-06'),
        ('morph_tok', 'nofreeze', 8, '1e-06'),
        ('morph_tok', 'nofreeze', 16, '1e-06'),
        ('morph_tok', 'nofreeze', 32, '1e-06'),
        ('morph_tok', 'nofreeze', 64, '1e-06')
    ]

    for bm in best_models:

        freeze = bm[1] == 'freeze'

        print('Mode: {}'.format(bm[0]))
        print('Freeze: {}'.format(freeze))
        print('Count: {}'.format(bm[2]))
        print('Batch size: {}'.format(args.batch_size))
        print('Learning rate: {}'.format(bm[3]))

        # Define poath to data
        inpath = str(Path('../../data/final').resolve())

        test_path = '{}{}sents_{:02d}_test_split.txt'.format(inpath, os.sep, bm[2])

        # Load val data
        print('Load validation data...')
        try:
            val_data = AffixDataset(test_path, bm[0])
        except FileNotFoundError:
            print('Bin not found.')
            continue

        # Sample negative examples for val data
        val_data.get_full_batch()
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_sents)

        mwf_predictor = MWFPredictor(freeze=freeze)

        # Define device
        device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        # Move model to CUDA
        mwf_predictor = mwf_predictor.to(device)

        print('Loading finetuned model weights from model_{}_{}_{:02d}_{}.torch...'.format(bm[0], bm[1], bm[2], bm[3]))
        mwf_predictor.load_state_dict(torch.load('trained_mwf/model_{}_{}_{:02d}_{}.torch'.format(bm[0], bm[1], bm[2], bm[3]), map_location=device))

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

        acc = len([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)

        with open('results_final/results_mwf_{}_{}.txt'.format(bm[0], bm[1]), 'a+') as f:
            f.write('{:.3f} & '.format(acc))


if __name__ == '__main__':
    main()
