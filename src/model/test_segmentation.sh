#!/bin/bash

python3.6 -u get_preds_hyp.py --batch_size 16 --cuda "$1" >> "output_final/output_preds_hyp.txt" 2>> "output_final/errors_preds_hyp.txt"
python3.6 -u get_preds_init.py --batch_size 16 --cuda "$1" >> "output_final/output_preds_init.txt" 2>> "output_final/errors_preds_init.txt"
python3.6 -u get_preds_tok.py --batch_size 16 --cuda "$1" >> "output_final/output_preds_tok.txt" 2>> "output_final/errors_preds_tok.txt"
python3.6 -u get_preds_proj.py --batch_size 16 --cuda "$1" >> "output_final/output_preds_proj.txt" 2>> "output_final/errors_preds_proj.txt"
