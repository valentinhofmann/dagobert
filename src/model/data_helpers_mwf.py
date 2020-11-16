from collections import Counter

import random
from random import Random
import numpy as np
import torch
from segmentation_tools import *
from torch.utils.data import Dataset
from transformers import BertTokenizer


class AffixDataset(Dataset):

    def __init__(self, filename, mode):

        self.mode = mode
        # self.my_random = Random(seed)

        # Initialize tokenizer
        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')

        self.affixes_true = list()

        self.bases = list()

        self.sents_1 = list()
        self.sents_2 = list()

        self.full_batch = list()

        with open(filename, 'r') as f:

            for l in f:

                sr, w, b, s_left, s, s_right = l.strip().split('|||')

                # Derive affixes
                d = derive(w, mode='morphemes')

                # Prefix (with trick)
                if len(d[0]) == 1 and len(d[2]) == 0:

                    # Skip sentences with '___'
                    if Counter(s.split())['___'] > 1:
                        continue

                    [s_1, s_2] = s.split('___')

                    self.sents_1.append(s_1)
                    self.sents_2.append(s_2)

                    self.affixes_true.append(d[0][0])
                    self.bases.append(b)

    def get_full_batch(self):

        # Initialize empty list
        self.full_batch = list()

        for a_true, b, s_1, s_2 in zip(self.affixes_true, self.bases, self.sents_1, self.sents_2):

            while True:
                a_false = random.choice(self.affixes_true)
                if a_false != a_true:
                    break

            if self.mode == 'bert_tok':

                w_true = self.tok.tokenize(a_true + b)
                w_false = self.tok.tokenize(a_false + b)

                s_true = ['[CLS]'] + self.tok.tokenize(s_1) + w_true + self.tok.tokenize(s_2) + ['[SEP]']
                s_false = ['[CLS]'] + self.tok.tokenize(s_1) + w_false + self.tok.tokenize(s_2) + ['[SEP]']

                idxes = (len(['[CLS]'] + self.tok.tokenize(s_1)), -len(self.tok.tokenize(s_2) + ['[SEP]']))

                self.full_batch.append((self.tok.convert_tokens_to_ids(s_true), self.tok.convert_tokens_to_ids(s_false), idxes))

            elif self.mode == 'morph_tok':

                w_true = self.tok.tokenize(a_true) + ['-'] + self.tok.tokenize(b)
                w_false = self.tok.tokenize(a_false) + ['-'] + self.tok.tokenize(b)

                s_true = ['[CLS]'] + self.tok.tokenize(s_1) + w_true + self.tok.tokenize(s_2) + ['[SEP]']
                s_false = ['[CLS]'] + self.tok.tokenize(s_1) + w_false + self.tok.tokenize(s_2) + ['[SEP]']

                idxes = (len(['[CLS]'] + self.tok.tokenize(s_1)), -len(self.tok.tokenize(s_2) + ['[SEP]']))

                self.full_batch.append((self.tok.convert_tokens_to_ids(s_true), self.tok.convert_tokens_to_ids(s_false), idxes))

    def __len__(self):
        return len(self.affixes_true)

    def __getitem__(self, idx):
        return self.full_batch[idx]


def collate_sents(batch):

    batch_size = len(batch)

    sents_true = [s_true for s_true, s_false, idxes in batch]
    sents_false = [s_false for s_true, s_false, idxes in batch]
    w_idxes = [idxes for s_true, s_false, idxes in batch] * 2

    max_len = max(len(s) for s in sents_true + sents_false)
    sents_pad = np.zeros((2 * batch_size, max_len))
    masks_pad = np.zeros((2 * batch_size, max_len))
    segs_pad = np.zeros((2 * batch_size, max_len))

    idxes = list()

    for i, s in enumerate(sents_true + sents_false):

        pad_len = max_len - len(s)

        idxes.append((w_idxes[i][0], w_idxes[i][1] - pad_len))

        sents_pad[i, :len(s)] = s
        masks_pad[i, :len(s)] = 1

    labels = len(batch) * [1] + len(batch) * [0]

    return torch.tensor(sents_pad).long(), torch.tensor(masks_pad).long(), torch.tensor(segs_pad).long(), torch.tensor(labels).float(), idxes
