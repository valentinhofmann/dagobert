#!/bin/bash

python3.6 -u get_preds_bert.py --batch_size 16 --cuda "$1" --finetuned >> "output_final/output_preds_finetuned.txt" 2>> "output_final/errors_preds_finetuned.txt"
python3.6 -u get_preds_bert_freeze.py --batch_size 16 --cuda "$1" >> "output_final/output_preds_freeze.txt" 2>> "output_final/errors_preds_freeze.txt"
python3.6 -u get_preds_lstm.py --batch_size 64 --cuda "$1" >> "output_final/output_preds_lstm.txt" 2>> "output_final/errors_preds_lstm.txt"

python3.6 -u random_model.py --mode pfx --batch_size 16 --lexicon shared >> "output_final/output_random_pfx.txt" 2>> "output_final/errors_random_pfx.txt"
python3.6 -u random_model.py --mode sfx --batch_size 16 --lexicon shared >> "output_final/output_random_sfx.txt" 2>> "output_final/errors_random_sfx.txt"
python3.6 -u random_model.py --mode both --batch_size 16 --lexicon shared >> "output_final/output_random_both.txt" 2>> "output_final/errors_random_both.txt"
python3.6 -u random_model.py --mode pfx --batch_size 16 --lexicon split >> "output_final/output_random_pfx.txt" 2>> "output_final/errors_random_pfx.txt"
python3.6 -u random_model.py --mode sfx --batch_size 16 --lexicon split >> "output_final/output_random_sfx.txt" 2>> "output_final/errors_random_sfx.txt"
python3.6 -u random_model.py --mode both --batch_size 16 --lexicon split >> "output_final/output_random_both.txt" 2>> "output_final/errors_random_both.txt"
