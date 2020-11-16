#!/bin/bash


for lr in 0.000001 0.000003 0.00001 0.00003
do
	python3.6 -u finetuning_vyl.py --lexicon shared --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_vyl/output_bert_vyl_shared.txt" 2>> "output_vyl/errors_bert_vyl_shared.txt"
	python3.6 -u finetuning_vyl.py --lexicon split --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_vyl/output_bert_vyl_split.txt" 2>> "output_vyl/errors_bert_vyl_split.txt"
done

for lr in 0.0001 0.0003 0.001 0.003
do
	python3.6 -u lstm_encoder_vyl.py --lexicon shared --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_vyl/output_lstm_vyl_shared.txt" 2>> "output_vyl/errors_lstm_vyl_shared.txt"
	python3.6 -u lstm_encoder_vyl.py --lexicon split --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_vyl/output_lstm_vyl_split.txt" 2>> "output_vyl/errors_lstm_vyl_split.txt"
done
