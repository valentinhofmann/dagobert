import os
import re
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
import pycld2 as cld2
from nltk import sent_tokenize, ngrams
from preprocessing import clean_sents
from segmentation_tools import derive

# Load user set for removal
spammers = set()

with open(str(Path(__file__).parent.parent.parent / 'data/external/bot_list.txt'), 'r') as file:
    for line in file:
        spammers.add(line.strip().lower())

with open(str(Path(__file__).parent.parent.parent / 'data/external/spammer_list.txt'), 'r') as file:
    for line in file:
        spammers.add(line.strip().lower())


# Define Reddit file class
class RedditFile:
    def __init__(self, infile, compression):
        self.infile = infile
        self.month = re.findall(r'\d{4}-\d{2}', os.path.basename(infile))[0]
        self.compression = compression

    # Define method to extract sentence triples from file
    def get_sents(self, outpath, size=5000, spammers=spammers):

        with open('{}{}sents_{}.txt'.format(outpath, os.sep, self.month), 'w', encoding='utf-8') as f:

            for c in pd.read_json(self.infile, compression=self.compression, lines=True, chunksize=size, dtype=False):

                for r in c.itertuples(index=False):

                    # Define authors and bodies to be skipped
                    skip_author = spammers.union({'[deleted]'})
                    skip_body = {'', '[deleted]', '[removed]'}

                    if (r.author not in skip_author) and (r.body not in skip_body):

                        # Skip posts with quotes
                        if escape('>') in r.body:
                            continue

                        if '|||' in r.body:
                            continue

                        # Tokenize post into sentences
                        sents = sent_tokenize(r.body)

                        # Clean sentences
                        sents = clean_sents(sents)

                        # Skip posts with links
                        if any('http' in s for s in sents):
                            continue

                        # Filter out posts with less than three sentences
                        if len(sents) < 3:
                            continue

                        # Filter out triples with sentences of less than 10 words
                        triples = [t for t in ngrams(sents, 3) if not any(len([w for w in s.split() if w.isalpha()]) < 10 for s in t)]

                        if len(triples) == 0:
                            continue

                        for t in triples:

                            for w in t[1].split():

                                # Derive word
                                d = derive(w, mode='roots')

                                # Check if word can be morphologically segmented
                                if d != w:

                                    if (w in t[0].split()) or (w in t[2].split()) or Counter(t[1])[w] > 1:
                                        continue

                                    if (cld2.detect(t[0])[2][0][1] != 'en') or (cld2.detect(t[1])[2][0][1] != 'en') or (cld2.detect(t[2])[2][0][1] != 'en'):
                                        continue

                                    f.write(r.subreddit + '|||' + w + '|||' + d + '|||' + '|||'.join(t) + '\n')
