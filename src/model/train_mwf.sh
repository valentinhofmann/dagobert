#!/bin/bash


for count in 01 02 04 08 16 32 64
do
	for lr in 0.000001 0.000003 0.00001 0.00003
	do
		python3.6 -u mwf.py --count $count --mode bert_tok --lexicon split --batch_size 8 --lr $lr --n_epochs 8 --cuda "$1" >> "output_mwf/output_bert_tok_nofreeze_${count}.txt" 2>> "output_mwf/errors_bert_tok_nofreeze_${count}.txt"
		python3.6 -u mwf.py --count $count --mode morph_tok --lexicon split --batch_size 8 --lr $lr --n_epochs 8 --cuda "$1" >> "output_mwf/output_morph_tok_nofreeze_${count}.txt" 2>> "output_mwf/errors_morph_tok_nofreeze_${count}.txt"
	done
	
	for lr in 0.0001 0.0003 0.001 0.003
	do
		python3.6 -u mwf.py --count $count --mode bert_tok --lexicon split --batch_size 8 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_mwf/output_bert_tok_freeze_${count}.txt" 2>> "output_mwf/errors_bert_tok_freeze_${count}.txt"
		python3.6 -u mwf.py --count $count --mode morph_tok --lexicon split --batch_size 8 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_mwf/output_morph_tok_freeze_${count}.txt" 2>> "output_mwf/errors_morph_tok_freeze_${count}.txt"
	done
done
