import argparse
import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from data_helpers import *
from evaluation import *
import random
from model import AffixPredictor
from segmentation_tools import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from finetuning import test_single, test_both


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--finetuned', default=False, action='store_true', help='Using finetuned BERT model.')

    args = parser.parse_args()

    best_models = [
        ('pfx', 1, 'shared', '1e-05'),
        ('pfx', 2, 'shared', '3e-05'),
        ('pfx', 4, 'shared', '3e-05'),
        ('pfx', 8, 'shared', '3e-05'),
        ('pfx', 16, 'shared', '1e-05'),
        ('pfx', 32, 'shared', '3e-06'),
        ('pfx', 64, 'shared', '3e-06'),

        ('pfx', 1, 'split', '3e-06'),
        ('pfx', 2, 'split', '1e-05'),
        ('pfx', 4, 'split', '3e-06'),
        ('pfx', 8, 'split', '1e-06'),
        ('pfx', 16, 'split', '3e-06'),
        ('pfx', 32, 'split', '1e-06'),
        ('pfx', 64, 'split', '1e-06'),

        ('sfx', 1, 'shared', '3e-05'),
        ('sfx', 2, 'shared', '3e-05'),
        ('sfx', 4, 'shared', '3e-05'),
        ('sfx', 8, 'shared', '3e-05'),
        ('sfx', 16, 'shared', '1e-05'),
        ('sfx', 32, 'shared', '1e-05'),
        ('sfx', 64, 'shared', '3e-06'),

        ('sfx', 1, 'split', '3e-05'),
        ('sfx', 2, 'split', '1e-05'),
        ('sfx', 4, 'split', '3e-06'),
        ('sfx', 8, 'split', '1e-06'),
        ('sfx', 16, 'split', '1e-06'),
        ('sfx', 32, 'split', '1e-06'),
        ('sfx', 64, 'split', '1e-06'),

        ('both', 1, 'shared', '1e-05'),
        ('both', 2, 'shared', '3e-05'),
        ('both', 4, 'shared', '3e-05'),
        ('both', 8, 'shared', '3e-05'),
        ('both', 16, 'shared', '1e-05'),
        ('both', 32, 'shared', '3e-05'),
        ('both', 64, 'shared', '1e-05'),

        ('both', 1, 'split', '1e-05'),
        ('both', 2, 'split', '3e-06'),
        ('both', 4, 'split', '3e-06'),
        ('both', 8, 'split', '1e-06'),
        ('both', 16, 'split', '1e-06'),
        ('both', 32, 'split', '1e-06'),
        ('both', 64, 'split', '1e-06')
    ]

    for bm in best_models:

        print('Mode: {}'.format(bm[0]))
        print('Count: {}'.format(bm[1]))
        print('Lexicon setting: {}'.format(bm[2]))
        print('Learning rate: {}'.format(bm[3]))

        print('Batch size: {}'.format(args.batch_size))
        print('Finetuned: {}'.format(args.finetuned))

        # Define path to data
        inpath = str(Path('../../data/final').resolve())

        if bm[2] == 'shared':
            test_path = '{}{}sents_{:02d}_test.txt'.format(inpath, os.sep, bm[1])

        elif bm[2] == 'split':
            test_path = '{}{}sents_{:02d}_test_split.txt'.format(inpath, os.sep, bm[1])

        # Initialize val loader
        print('Load validation data...')
        try:
            test_data = AffixDataset(test_path, bm[0])
        except FileNotFoundError:
            print('Bin not found.')
            continue

        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_sents)

        tok = BertTokenizer.from_pretrained('bert-base-uncased')

        # Define device
        device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        # Initialize model
        affix_predictor = AffixPredictor(bm[0], freeze=False)

        # Load finetuned model weights
        if args.finetuned:
            print('Loading finetuned model weights from model_bert_{}_{}_{:02d}_{}.torch...'.format(
                bm[2], bm[0], bm[1], bm[3]))
            affix_predictor.load_state_dict(torch.load('trained_main/model_bert_{}_{}_{:02d}_{}.torch'.format(
                bm[2], bm[0], bm[1], bm[3]), map_location=device))

        # Move model to CUDA
        affix_predictor = affix_predictor.to(device)

        if bm[0] == 'pfx' or bm[0] == 'sfx':
            mrr_micro, mrr_macro_dict = test_single(test_loader, affix_predictor, bm[0], args.cuda)
        elif bm[0] == 'both':
            mrr_micro, mrr_macro_dict = test_both(test_loader, affix_predictor, args.cuda)

        if args.finetuned:
            with open('results_final/results_bert_{}_{}_finetuned_macro.txt'.format(bm[2], bm[0]), 'a+') as f:
                f.write('{:.3f} & '.format(np.mean(list(mrr_macro_dict.values()))))
            with open('results_final/results_bert_{}_{}_finetuned_micro.txt'.format(bm[2], bm[0]), 'a+') as f:
                f.write('{:.3f} & '.format(mrr_micro))
        else:
            with open('results_final/results_bert_{}_{}_basic_macro.txt'.format(bm[2], bm[0]), 'a+') as f:
                f.write('{:.3f} & '.format(np.mean(list(mrr_macro_dict.values()))))
            with open('results_final/results_bert_{}_{}_basic_micro.txt'.format(bm[2], bm[0]), 'a+') as f:
                f.write('{:.3f} & '.format(mrr_micro))


if __name__ == '__main__':
    main()
