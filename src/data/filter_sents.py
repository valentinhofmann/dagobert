import glob
import os
import pickle
from pathlib import Path
from segmentation_tools import *

# Load counter of derivatives
with open('d_counter.p', 'rb') as f:
    d_counter = pickle.load(f)

# Define path to files
inpath = str(Path('../../data/sents'))

# Create list of files
infile_list = glob.glob("%s%s*" % (inpath, os.sep))

# Define path for output
outpath = str(Path('../../data/filtered'))

for i, j in zip(range(0, 7), range(1, 8)):

    s, e = 2 ** i, 2 ** j

    with open('{}{}sents_filtered_{:02d}.txt'.format(outpath, os.sep, s), 'w', encoding='utf-8') as outfile:

        # Loop over months
        for infile in infile_list:

            with open(infile, 'r', encoding='utf-8') as f:

                # Loop over examples
                for l in f:

                    # Catch cases with '|||' in text
                    try:
                        sr, d, b, s_1, s_2, s_3 = l.strip().split('|||')
                    except ValueError:
                        continue

                    # Only include derivatives in frequency band
                    if s <= d_counter[d] < e:

                        der = derive(d, mode='morphemes')

                        # Skip derivatives with 'abil'
                        if 'abil' in der[2]:
                            continue

                        # Keep derivatives with at most one prefix and one suffix
                        if len(der[0]) > 1 or len(der[2]) > 1:
                            continue

                        # Skip long sentences
                        if len(s_1.strip().split()) > 100 or len(s_2.strip().split()) > 100 or len(s_3.strip().split()) > 100:
                            continue

                        # Skip sentences with '___'
                        if '___' in s_1 or '___' in s_2 or '___' in s_3:
                            continue

                        # Replace derivative with '___' and write to file
                        else:
                            s_2 = ' '.join(['___' if w == d else w for w in s_2.strip().split()])
                            outfile.write('|||'.join([sr, d, b, s_1, s_2, s_3]) + '\n')
