from collections import Counter

import numpy as np
import torch
from segmentation_tools import *
from torch.utils.data import Dataset
from transformers import BertTokenizer


class AffixDataset(Dataset):

    def __init__(self, filename, mode):

        # Define which affix type to extract
        self.mode = mode

        # Initialize tokenizer
        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')

        self.sents = list()
        self.idxes_mask = list()
        self.labels = list()

        with open(filename, 'r') as f:

            for l in f:

                sr, w, b, s_left, s, s_right = l.strip().split('|||')

                # Derive affixes
                d = derive(w, mode='morphemes')

                # Prefix (with trick)
                if self.mode == 'pfx' and len(d[0]) == 1 and len(d[2]) == 0:

                    # Skip sentences with '___'
                    if Counter(s.split())['___'] > 1:
                        continue

                    [s_1, s_2] = s.split('___')

                    # Tokenize sentence and add mask token
                    s = ['[CLS]'] + self.tok.tokenize(s_1) + ['[MASK]', '-'] + self.tok.tokenize(b) + self.tok.tokenize(s_2) + ['[SEP]']

                    # Store index of mask token
                    self.idxes_mask.append(s.index('[MASK]'))

                    # Encode sentence
                    s = self.tok.convert_tokens_to_ids(s)

                    # Store tokenized sentence and label
                    self.sents.append(s)
                    self.labels.append(self.tok.convert_tokens_to_ids(d[0][0]))

                # Suffix
                elif self.mode == 'sfx' and len(d[0]) == 0 and len(d[2]) == 1:

                    # Skip sentences with '___'
                    if Counter(s.split())['___'] > 1:
                        continue

                    [s_1, s_2] = s.split('___')

                    # Tokenize sentence and add mask token
                    s = ['[CLS]'] + self.tok.tokenize(s_1) + self.tok.tokenize(b) + ['[MASK]'] + self.tok.tokenize(s_2) + ['[SEP]']

                    # Store index of mask token
                    self.idxes_mask.append(s.index('[MASK]'))

                    # Encode sentence
                    s = self.tok.convert_tokens_to_ids(s)

                    # Store tokenized sentence and label
                    self.sents.append(s)
                    self.labels.append(self.tok.convert_tokens_to_ids('##' + d[2][0]))

                # Both
                elif self.mode == 'both' and len(d[0]) == 1 and len(d[2]) == 1:

                    # Skip sentences with '___'
                    if Counter(s.split())['___'] > 1:
                        continue

                    [s_1, s_2] = s.split('___')

                    # Tokenize sentence and add mask tokens
                    s = ['[CLS]'] + self.tok.tokenize(s_1) + ['[MASK]', '-'] + self.tok.tokenize(b) + ['[MASK]'] + self.tok.tokenize(s_2) + ['[SEP]']

                    # Store index of mask token
                    self.idxes_mask.append([i for i in range(len(s)) if s[i] == '[MASK]'])

                    # Encode sentence
                    s = self.tok.convert_tokens_to_ids(s)

                    # Store tokenized sentence and label
                    self.sents.append(s)
                    self.labels.append([self.tok.convert_tokens_to_ids(d[0][0]), self.tok.convert_tokens_to_ids('##' + d[2][0])])

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
