#!/bin/bash

python3.6 -u get_preds_mwf.py --batch_size 8 --cuda "$1"  >> "output_final/output_mwf.txt" 2>> "output_final/errors_mwf.txt"
