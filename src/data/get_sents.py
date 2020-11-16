import os
import re
import sys
from pathlib import Path

from reddit_tools import RedditFile

sys.stdout.flush()

# Read name of input file
infile = str(sys.argv[1])

# Define path for ouput
outpath = str(Path('../../data/sents'))

# Process data
print('Processing file {}...'.format(infile))
reddit_file = RedditFile(infile, re.findall(r'\.(\w+)', os.path.basename(infile))[0])
reddit_file.get_sents(outpath)
print('Processing of file {} finished.'.format(infile))
