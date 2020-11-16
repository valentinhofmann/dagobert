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

        # Loop over examples
        with open('{}{}sents_filtered_{:02d}.txt'.format(inpath, os.sep, count), 'r') as in_f:

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
            with open('{}{}sents_{:02d}_{}.txt'.format(outpath, os.sep, count, name), 'w') as out_f:
                for s in type:
                    out_f.write(s + '\n')


if __name__ == '__main__':

    ratio_train = float(sys.argv[1])
    ratio_dev = float(sys.argv[2])

    main(ratio_train, ratio_dev)
