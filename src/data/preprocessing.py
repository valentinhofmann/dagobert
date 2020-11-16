import re
from html import unescape
from pathlib import Path
from nltk import word_tokenize
from string import punctuation

# Initialize prefix list
prefix_list = []
with open(str(Path(__file__).parent.parent.parent / 'data/external/bert_prefixes.txt'), 'r') as file:
    for line in file:
        prefix_list.append(line.strip().lower())
px = tuple(prefix_list)

# Initialize suffix list
suffix_list = []
with open(str(Path(__file__).parent.parent.parent / 'data/external/bert_suffixes.txt'), 'r') as file:
    for line in file:
        suffix_list.append(line.strip().lower())
sx = tuple(suffix_list)


# Define function to clean posts
def clean_sents(sents):

    # Tokenize and remove HTML formatting
    sents = [word_tokenize(unescape(s.lower().strip())) for s in sents]

    # Remove leading and trailing punctuation around words except for apostrophe
    sents = [[w.strip(re.sub(r'\'', '', punctuation)) if w not in punctuation else w for w in s] for s in sents]

    # Remove hyphens from morphologically complex words
    sents = [[re.sub(r'\-', '', w) if w.startswith(px) or w.endswith(sx) else w for w in s] for s in sents]

    # Remove special characters
    sents = [[re.sub('[^a-z0-9{}]'.format(punctuation), '', w) for w in s] for s in sents]

    # Remove embedded links
    sents = [re.sub('\[?\s(.*)\s\]\s\(\shttp.*\s\)', r'\1', ' '.join(s)) for s in sents]

    return [' '.join([w.strip() for w in s.split()]) for s in sents]
