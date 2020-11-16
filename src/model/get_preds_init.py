import argparse
import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from data_helpers_init import *
from evaluation import *
import random
from model import AffixPredictor
from segmentation_tools import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')

    args = parser.parse_args()

    best_models = [
        ('pfx', 1),
        ('pfx', 2),
        ('pfx', 4),
        ('pfx', 8),
        ('pfx', 16),
        ('pfx', 32),
        ('pfx', 64)
    ]

    for bm in best_models:

        print('Mode: {}'.format(bm[0]))
        print('Count: {}'.format(bm[1]))

        print('Batch size: {}'.format(args.batch_size))

        # Define path to data
        inpath = str(Path('../../data/final').resolve())

        test_path = '{}{}sents_{:02d}_test.txt'.format(inpath, os.sep, bm[1])

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

        # Move model to CUDA
        affix_predictor = affix_predictor.to(device)

        mode2afx = {'pfx': 'prefixes', 'sfx': 'suffixes'}

        # Initialize affix list
        afxes = []
        with open(str(Path('../../data/external/bert_{}.txt'.format(mode2afx[bm[0]])).resolve()), 'r') as f:
            for l in f:
                if l.strip() == '' or l.strip() == 'abil':
                    continue
                afxes.append(l.strip().lower())

        if bm[0] == 'pfx':
            idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(afxes)).to(device))

        elif bm[0] == 'sfx':
            idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(['##' + a for a in afxes])).to(device))

        print('Evaluating model...')

        y_true = list()
        y_pred = list()
        bases = list()

        affix_predictor.eval()

        with torch.no_grad():

            for batch in test_loader:

                sents, masks, segs, idxes_mask, labels = batch

                if bm[0] == 'pfx':
                    # Prefix: base one step to the right
                    bases.extend(sents[torch.arange(sents.size(0)), idxes_mask + 1].tolist())
                elif bm[0] == 'sfx':
                    # Suffix: base one step to the left
                    bases.extend(sents[torch.arange(sents.size(0)), idxes_mask - 1].tolist())

                sents, masks, segs, idxes_mask = sents.to(device), masks.to(device), segs.to(device), idxes_mask.to(device)

                # Forward pass
                output = affix_predictor(sents, masks, segs, idxes_mask)

                # Filter affixes
                output_afx = torch.index_select(output, -1, idxes_afx)

                # Rank predictions
                vals_afx, preds_afx = torch.topk(output_afx, k=output_afx.size(-1), dim=-1)

                labels = labels.to(device)

                # Store labels and predictions
                y_true.extend([l.item() for l in labels])
                y_pred.extend([[idxes_afx[p].item() for p in list(l)] for l in preds_afx])

                # Delete tensors to free memory
                del sents, masks, segs, idxes_mask, labels, output

        with open('results_final/results_init_macro.txt', 'a+') as f:
            f.write('{:.3f} & '.format(np.mean(list(mrr_macro(y_true, y_pred, 10).values()))))

        with open('results_final/results_init_micro.txt', 'a+') as f:
            f.write('{:.3f} & '.format(mrr_micro(y_true, y_pred, 10)))


if __name__ == '__main__':
    main()
