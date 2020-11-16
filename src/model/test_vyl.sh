#!/bin/bash

python3.6 -u get_preds_bert_vyl.py --batch_size 16 --cuda "$1" --finetuned --lexicon shared --lr 3e-06 >> "output_final/output_preds_vyl_shared_finetuned.txt" 2>> "output_final/errors_preds_vyl_shared_finetuned.txt"
python3.6 -u get_preds_bert_vyl.py --batch_size 16 --cuda "$1" --finetuned --lexicon split --lr 1e-05 >> "output_final/output_preds_vyl_split_finetuned.txt" 2>> "output_final/errors_preds_vyl_split_finetuned.txt"

python3.6 -u get_preds_lstm_vyl.py --batch_size 64 --cuda "$1" --lexicon shared --lr 1e-04 >> "output_final/output_preds_lstm_vyl_shared.txt" 2>> "output_final/errors_preds_lstm_vyl_shared.txt"
python3.6 -u get_preds_lstm_vyl.py --batch_size 64 --cuda "$1" --lexicon split --lr 1e-04 >> "output_final/output_preds_lstm_vyl_split.txt" 2>> "output_final/errors_preds_lstm_vyl_split.txt"
