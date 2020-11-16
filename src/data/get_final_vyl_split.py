import math
import re
import os
import random
import sys
from pathlib import Path


def main(ratio_train, ratio_dev):

    random.seed(1)

    # Define input path
    inpath = str(Path('../../data/sents').resolve())

    # Define output path
    outpath = str(Path('../../data/final').resolve())

    sents = list()
    bases = set()

    # Loop over examples
    with open('{}{}out.log2'.format(inpath, os.sep), 'r') as in_f:

        for l in in_f:

            sents.append(l.strip())

            b, d, f, c, afx, s_1, s_2 = l.strip().split('|||')

            bases.add(re.sub('[^a-z]', '', b))

    bases = list(bases)

    n_train = math.floor(ratio_train * len(bases))
    n_dev = math.floor(ratio_dev * len(bases))

    # Shuffle bases
    random.shuffle(bases)

    # Select corresponding share of bases
    bases_train, bases_dev, bases_test = bases[:n_train], bases[n_train:n_train+n_dev], bases[n_train+n_dev:]

    for type, name in zip([bases_train, bases_dev, bases_test], ['train', 'dev', 'test']):

        with open('{}{}sents_vyl_{}_split.txt'.format(outpath, os.sep, name), 'w') as out_f:

            for sent in sents:

                b, d, f, c, afx, s_1, s_2 = sent.strip().split('|||')

                if re.sub('[^a-z]', '', b) in type:

                    out_f.write(sent + '\n')


if __name__ == '__main__':

    ratio_train = float(sys.argv[1])
    ratio_dev = float(sys.argv[2])

    main(ratio_train, ratio_dev)
