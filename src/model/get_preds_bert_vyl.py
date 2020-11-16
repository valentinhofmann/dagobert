import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from data_helpers_vyl import *
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
    parser.add_argument('--finetuned', default=False, action='store_true', help='Using finetuned BERT model.')
    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting')
    parser.add_argument('--lr', default=None, type=str, required=True, help='Learning rate.')

    args = parser.parse_args()

    print('Lexicon setting: {}'.format(args.lexicon))
    print('Batch size: {}'.format(args.batch_size))
    print('Finetuned: {}'.format(args.finetuned))
    print('Learning rate: {}'.format(args.lr))

    # Define path to data
    inpath = str(Path('../../data/final').resolve())

    if args.lexicon == 'shared':
        test_path = '{}{}sents_vyl_test.txt'.format(inpath, os.sep)

    elif args.lexicon == 'split':
        test_path = '{}{}sents_vyl_test_split.txt'.format(inpath, os.sep)

    # Initialize val loader
    print('Load validation data...')
    test_data = AffixDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_sents)

    tok = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define device
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    # Initialize model
    affix_predictor = AffixPredictor('sfx', freeze=False)

    # Load finetuned model weights
    if args.finetuned:
        print('Loading finetuned model weights from model_bert_vyl_{}_{}.torch...'.format(args.lexicon, args.lr))
        affix_predictor.load_state_dict(torch.load('trained_vyl/model_bert_vyl_{}_{}.torch'.format(args.lexicon, args.lr), map_location=device))

    # Move model to CUDA
    affix_predictor = affix_predictor.to(device)

    # Initialize affix list
    afxes = []
    with open(str(Path('../../data/external/affixes_vyl.txt').resolve()), 'r') as f:
        for l in f:
            # Exclude affixes not in BERT vocabulary
            if l.strip() == '' or l.strip() in {'NULL', 'orium', 'tude'}:
                continue
            afxes.append(l.strip().lower())

    # Add affixes not in BERT as unused tokens
    afxes = ['[unused96]', '[unused97]', '[unused98]'] + ['##' + afx for afx in afxes]
    idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(afxes)).to(device))

    print('Evaluating model...')

    y_true = list()
    y_pred = list()

    affix_predictor.eval()

    with torch.no_grad():

        for batch in test_loader:

            sents, masks, segs, idxes_mask, labels = batch

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
            y_pred.extend([[idxes_afx[p].item() for p in list(l)][0] for l in preds_afx])

            # Delete tensors to free memory
            del sents, masks, segs, idxes_mask, labels, output

    acc = len([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)

    if args.finetuned:
        with open('results_final/results_vyl_bert_finetuned.txt', 'a+') as f:
            f.write('{:.3f} & '.format(acc))
    else:
        with open('results_final/results_vyl_bert_basic.txt', 'a+') as f:
            f.write('{:.3f} & '.format(acc))


if __name__ == '__main__':
    main()
