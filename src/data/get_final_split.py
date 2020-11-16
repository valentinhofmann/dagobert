import math
import os
import random
import sys
from pathlib import Path


def main(ratio_train, ratio_dev):

    random.seed(1)

    # Define input path
    inpath = str(Path('../../data/filtered').resolve())

    # Define output path
    outpath = str(Path('../../data/final').resolve())

    for count in [1, 2, 4, 8, 16, 32, 64]:

        sents = list()
        bases = set()

        # Loop over examples
        with open('{}{}sents_filtered_{:02d}.txt'.format(inpath, os.sep, count), 'r') as in_f:

            for l in in_f:

                sents.append(l.strip())

                sr, w, b, s_left, s, s_right = l.strip().split('|||')

                bases.add(b)

        bases = list(bases)

        n_train = math.floor(ratio_train * len(bases))
        n_dev = math.floor(ratio_dev * len(bases))

        # Shuffle bases
        random.shuffle(bases)

        # Select corresponding share of bases
        bases_train, bases_dev, bases_test = bases[:n_train], bases[n_train:n_train+n_dev], bases[n_train+n_dev:]

        for type, name in zip([bases_train, bases_dev, bases_test], ['train', 'dev', 'test']):

            with open('{}{}sents_{:02d}_{}_split.txt'.format(outpath, os.sep, count, name), 'w') as out_f:

                for sent in sents:

                    sr, w, b, s_left, s, s_right = sent.strip().split('|||')

                    if b in type:

                        out_f.write(sent + '\n')


if __name__ == '__main__':

    ratio_train = float(sys.argv[1])
    ratio_dev = float(sys.argv[2])

    main(ratio_train, ratio_dev)
