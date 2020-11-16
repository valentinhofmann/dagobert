import math
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


    # Loop over examples
    with open('{}{}out.log2'.format(inpath, os.sep), 'r') as in_f:

        sents = list()

        for l in in_f:

            sents.append(l.strip())

    n_train = math.floor(ratio_train * len(sents))
    n_dev = math.floor(ratio_dev * len(sents))

    # Shuffle split
    random.shuffle(sents)

    # Select corresponding share of sentences
    sents_train, sents_dev, sents_test = sents[:n_train], sents[n_train:n_train+n_dev], sents[n_train+n_dev:]

    for type, name in zip([sents_train, sents_dev, sents_test], ['train', 'dev', 'test']):
        with open('{}{}sents_vyl_{}.txt'.format(outpath, os.sep, name), 'w') as out_f:
            for s in type:
                out_f.write(s + '\n')


if __name__ == '__main__':

    ratio_train = float(sys.argv[1])
    ratio_dev = float(sys.argv[2])

    main(ratio_train, ratio_dev)
