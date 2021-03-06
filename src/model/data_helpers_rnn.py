from collections import Counter

import numpy as np
import torch
from segmentation_tools import *
from torch.utils.data import Dataset


class AffixDataset(Dataset):

    def __init__(self, filename, mode):

        # Create counter for character vocabulary
        self.c_vocab = Counter()

        # Define which affix type to extract
        self.mode = mode

        self.sents_1 = list()
        self.sents_2 = list()
        self.bases = list()
        self.labels = list()

        with open(filename, 'r') as f:

            for l in f:

                sr, w, b, s_left, s, s_right = l.strip().split('|||')

                # Derive affixes
                d = derive(w, mode='morphemes')

                # Prefix
                if self.mode == 'pfx' and len(d[0]) == 1 and len(d[2]) == 0:

                    # Skip sentences with '___'
                    if Counter(s.split())['___'] > 1:
                        continue

                    [s_1, s_2] = s.split('___')

                    # Store left and right contexts
                    self.sents_1.append(['<s>'] + s_1.split())
                    self.sents_2.append(s_2.split() + ['</s>'])

                    # Store base
                    self.bases.append(['<s>'] + [c for c in b] + ['</s>'])

                    # Update character vocabulary
                    self.c_vocab.update([c for c in b])
                    self.c_vocab.update(['<s>', '</s>'])

                    # Store label
                    self.labels.append(d[0][0])

                # Suffix
                elif self.mode == 'sfx' and len(d[0]) == 0 and len(d[2]) == 1:

                    # Skip sentences with '___'
                    if Counter(s.split())['___'] > 1:
                        continue

                    [s_1, s_2] = s.split('___')

                    # Store left and right contexts
                    self.sents_1.append(['<s>'] + s_1.split())
                    self.sents_2.append(s_2.split() + ['</s>'])

                    # Store base
                    self.bases.append(['<s>'] + [c for c in b] + ['</s>'])

                    # Update character vocabulary
                    self.c_vocab.update([c for c in b])
                    self.c_vocab.update(['<s>', '</s>'])

                    # Store label
                    self.labels.append(d[2][0])

                # Both
                elif self.mode == 'both' and len(d[0]) == 1 and len(d[2]) == 1:

                    # Skip sentences with '___'
                    if Counter(s.split())['___'] > 1:
                        continue

                    [s_1, s_2] = s.split('___')

                    # Store left and right contexts
                    self.sents_1.append(['<s>'] + s_1.split())
                    self.sents_2.append(s_2.split() + ['</s>'])

                    # Store base
                    self.bases.append(['<s>'] + [c for c in b] + ['</s>'])

                    # Update character vocabulary
                    self.c_vocab.update([c for c in b])
                    self.c_vocab.update(['<s>', '</s>'])

                    # Store label
                    self.labels.append((d[0][0], d[2][0]))

        # Define mapping from characters to integers
        self.c2id = {c: i + 2 for i, c in enumerate(c for c, count in self.c_vocab.most_common())}

        # Define mapping to integers for labels
        self.l2id = {l: i for i, l in enumerate(l for l, count in Counter(self.labels).most_common())}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Select sentence contexts, base, and label
        s_1 = self.sents_1[idx]
        s_2 = self.sents_2[idx]
        b = self.bases[idx]
        l = self.labels[idx]

        return s_1, s_2, b, l


class SentsCollator():

    def __init__(self, w2id, c2id, l2id):

        self.w2id = w2id
        self.c2id = c2id
        self.l2id = l2id

    def __call__(self, batch):

        batch_size = len(batch)

        sents_1 = [s_1 for s_1, s_2, b, l in batch]
        sents_2 = [s_2 for s_1, s_2, b, l in batch]
        bases = [b for s_1, s_2, b, l in batch]
        labels = [l for s_1, s_2, b, l in batch]

        # Get maximum length of sentence contexts and bases
        max_s_1 = max(len(s) for s in sents_1)
        max_s_2 = max(len(s) for s in sents_2)
        max_b = max(len(b) for b in bases)

        sents_1_pad = np.zeros((batch_size, max_s_1))
        sents_2_pad = np.zeros((batch_size, max_s_2))
        bases_pad = np.zeros((batch_size, max_b))

        for i, s in enumerate(sents_1):
            sents_1_pad[i, :len(s)] = [self.w2id[w] if w in self.w2id else 1 for w in s]

        for i, s in enumerate(sents_2):
            sents_2_pad[i, :len(s)] = [self.w2id[w] if w in self.w2id else 1 for w in s]

        for i, b in enumerate(bases):
            bases_pad[i, :len(b)] = [self.c2id[c] if c in self.c2id else 1 for c in b]

        labels = [self.l2id[l] if l in self.l2id else len(self.l2id) for l in labels]

        return torch.tensor(sents_1_pad).long(), torch.tensor(sents_2_pad).long(), torch.tensor(bases_pad).long(), torch.tensor(labels).long()
