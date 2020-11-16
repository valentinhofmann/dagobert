import re

import numpy as np
import torch
from segmentation_tools import *
from torch.utils.data import Dataset
from transformers import BertTokenizer


class AffixDataset(Dataset):

    def __init__(self, filename):

        # Initialize tokenizer
        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')

        self.sents = list()
        self.idxes_mask = list()
        self.labels = list()

        with open(filename, 'r') as f:

            for l in f:

                b, d, f, c, afx, s_1, s_2 = l.strip().split('|||')

                b, d, f, c, afx = [re.sub('[^a-zA-Z]', '', t) for t in [b, d, f, c, afx]]

                # Remove start and end of sentence tokens
                s_1, s_2 = ' '.join(s_1.strip().split()[1:]), ' '.join(s_2.strip().split()[:-1])

                # Tokenize sentence and add mask token
                s = ['[CLS]'] + self.tok.tokenize(s_1) + self.tok.tokenize(b) + ['[MASK]'] + self.tok.tokenize(s_2) + ['[SEP]']

                # Store index of mask token
                self.idxes_mask.append(s.index('[MASK]'))

                # Encode sentence
                s = self.tok.convert_tokens_to_ids(s)

                # Store tokenized sentence
                self.sents.append(s)

                # Use unused tokens for NULL affix, 'tude', and 'orium'
                if afx == 'NULL':
                    self.labels.append(97)
                elif afx == 'tude':
                    self.labels.append(98)
                elif afx == 'orium':
                    self.labels.append(99)
                # Convert all other suffixes to BERT ID
                else:
                    self.labels.append(self.tok.convert_tokens_to_ids('##' + afx))

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):

        # Select sentence, index if mask token, and label
        s = self.sents[idx]
        idx_mask = self.idxes_mask[idx]
        l = self.labels[idx]

        return s, idx_mask, l


def collate_sents(batch):
    batch_size = len(batch)

    sents = [s for s, idx_mask, l in batch]
    idxes_mask = [idx_mask for s, idx_mask, l in batch]
    labels = [l for s, idx_mask, l in batch]

    # Get maximum sentence length in batch
    max_len = max(len(s) for s in sents)

    sents_pad = np.zeros((batch_size, max_len))
    masks_pad = np.zeros((batch_size, max_len))
    segs_pad = np.zeros((batch_size, max_len))

    for i, s in enumerate(sents):
        sents_pad[i, :len(s)] = s
        masks_pad[i, :len(s)] = 1

    return torch.tensor(sents_pad).long(), torch.tensor(masks_pad).long(), torch.tensor(segs_pad).long(), torch.tensor(idxes_mask).long(), torch.tensor(labels).long()
