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

    args = parser.parse_args()

    models = [
        ('pfx', 1),
        ('pfx', 2),
        ('pfx', 4),
        ('pfx', 8),
        ('pfx', 16),
        ('pfx', 32),
        ('pfx', 64)
    ]

    for m in models:

        print('Mode: {}'.format(m[0]))
        print('Count: {}'.format(m[1]))

        print('Batch size: {}'.format(args.batch_size))

        # Define path to data
        inpath = str(Path('../../data/final').resolve())

        test_path = '{}{}sents_{:02d}_test.txt'.format(inpath, os.sep, m[1])

        # Initialize val loader
        print('Load validation data...')
        try:
            test_data = AffixDataset(test_path, m[0])
        except FileNotFoundError:
            print('Bin not found.')
            continue

        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_sents)

        tok = BertTokenizer.from_pretrained('bert-base-uncased')

        # Define device
        device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        # Initialize model
        affix_predictor = AffixPredictor(m[0], freeze=False)

        # Move model to CUDA
        affix_predictor = affix_predictor.to(device)

        mrr_micro, mrr_macro_dict = test_single(test_loader, affix_predictor, m[0], args.cuda)

        with open('results_final/results_hyp_macro.txt', 'a+') as f:
            f.write('{:.3f} & '.format(np.mean(list(mrr_macro_dict.values()))))
        with open('results_final/results_hyp_micro.txt', 'a+') as f:
            f.write('{:.3f} & '.format(mrr_micro))


if __name__ == '__main__':
    main()
