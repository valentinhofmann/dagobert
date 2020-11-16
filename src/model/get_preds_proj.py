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
from model_proj import AffixPredictor
from segmentation_tools import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel


def get_embs(sents, idxes_mask, projection_matrix, input_embs, tok):

    # Initialize tensor for embeddings
    embs = torch.zeros(sents.size(0), sents.size(1), 768)

    # Loop over sentences
    for i, (sent, idx_mask) in enumerate(zip(sents, idxes_mask)):

        # Convert indices to input embeddings
        s_1 = input_embs(sent[:idx_mask + 1])
        s_2 = input_embs(sent[idx_mask + 2:])

        b = tok.convert_ids_to_tokens(int(sent[idx_mask + 1]))

        if '##' + b in tok.vocab:
            b_emb = input_embs(torch.tensor(tok.convert_tokens_to_ids('##' + b))).unsqueeze(0)
        else:
            b_emb = torch.mm(input_embs(sent[idx_mask + 1]).unsqueeze(0), projection_matrix)

        s = torch.cat((s_1, b_emb, s_2))

        # Store sentence embeddings in tensor
        embs[i] = s

    return embs


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')

    args = parser.parse_args()

    # Load projection matrix
    with open('projection_matrix.p', 'rb') as f:
        projection_matrix = pickle.load(f)

    tok = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load BERT input embeddings
    bert = BertModel.from_pretrained('bert-base-uncased')
    input_embs = bert.get_input_embeddings()

    modes = [
        ('pfx', 1),
        ('pfx', 2),
        ('pfx', 4),
        ('pfx', 8),
        ('pfx', 16),
        ('pfx', 32),
        ('pfx', 64),
    ]

    for m in modes:

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

        # Define device
        device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        # Initialize model
        affix_predictor = AffixPredictor(m[0])

        # Move model to CUDA
        affix_predictor = affix_predictor.to(device)

        mode2afx = {'pfx': 'prefixes', 'sfx': 'suffixes'}

        # Initialize affix list
        afxes = []
        with open(str(Path('../../data/external/bert_{}.txt'.format(mode2afx[m[0]])).resolve()), 'r') as f:
            for l in f:
                if l.strip() == '' or l.strip() == 'abil':
                    continue
                afxes.append(l.strip().lower())

        if m[0] == 'pfx':
            idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(afxes)).to(device))

        elif m[0] == 'sfx':
            idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(['##' + a for a in afxes])).to(device))

        print('Evaluating model...')

        y_true = list()
        y_pred = list()
        bases = list()

        affix_predictor.eval()

        with torch.no_grad():

            for batch in test_loader:

                sents, masks, segs, idxes_mask, labels = batch

                if m[0] == 'pfx':
                    # Prefix: base one step to the right
                    bases.extend(sents[torch.arange(sents.size(0)), idxes_mask + 1].tolist())
                elif m[0] == 'sfx':
                    # Suffix: base one step to the left
                    bases.extend(sents[torch.arange(sents.size(0)), idxes_mask - 1].tolist())

                embs = get_embs(sents, idxes_mask, projection_matrix, input_embs, tok)

                embs, masks, segs, idxes_mask = embs.to(device), masks.to(device), segs.to(device), idxes_mask.to(device)

                # Forward pass
                output = affix_predictor(embs, masks, segs, idxes_mask)

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

        mrr_mic, mrr_mac_dict = mrr_micro(y_true, y_pred, 10), mrr_macro(y_true, y_pred, 10)
        mrr_mac = np.mean(list(mrr_mac_dict.values()))

        with open('results_final/results_proj_macro.txt', 'a+') as f:
            f.write('{:.3f} & '.format(mrr_mac))

        with open('results_final/results_proj_micro.txt', 'a+') as f:
            f.write('{:.3f} & '.format(mrr_mic))


if __name__ == '__main__':
    main()
