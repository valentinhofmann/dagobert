#!/bin/bash


for count in 01 02 04 08 16 32 64
do
	for lr in 0.000001 0.000003 0.00001 0.00003
	do
		python3.6 -u finetuning.py --count $count --mode pfx --lexicon shared --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_main/output_bert_shared_pfx_${count}.txt" 2>> "output_main/errors_bert_shared_pfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode sfx --lexicon shared --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_main/output_bert_shared_sfx_${count}.txt" 2>> "output_main/errors_bert_shared_sfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode both --lexicon shared --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_main/output_bert_shared_both_${count}.txt" 2>> "output_main/errors_bert_shared_both_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode pfx --lexicon split --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_main/output_bert_split_pfx_${count}.txt" 2>> "output_main/errors_bert_split_pfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode sfx --lexicon split --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_main/output_bert_split_sfx_${count}.txt" 2>> "output_main/errors_bert_split_sfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode both --lexicon split --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" >> "output_main/output_bert_split_both_${count}.txt" 2>> "output_main/errors_bert_split_both_${count}.txt"
	done
	
	for lr in 0.0001 0.0003 0.001 0.003
	do
		python3.6 -u finetuning.py --count $count --mode pfx --lexicon shared --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_main/output_bert_freeze_shared_pfx_${count}.txt" 2>> "output_main/errors_bert_freeze_shared_pfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode sfx --lexicon shared --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_main/output_bert_freeze_shared_sfx_${count}.txt" 2>> "output_main/errors_bert_freeze_shared_sfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode both --lexicon shared --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_main/output_bert_freeze_shared_both_${count}.txt" 2>> "output_main/errors_bert_freeze_shared_both_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode pfx --lexicon split --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_main/output_bert_freeze_split_pfx_${count}.txt" 2>> "output_main/errors_bert_freeze_split_pfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode sfx --lexicon split --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_main/output_bert_freeze_split_sfx_${count}.txt" 2>> "output_main/errors_bert_freeze_split_sfx_${count}.txt"
		python3.6 -u finetuning.py --count $count --mode both --lexicon split --batch_size 16 --lr $lr --n_epochs 8 --cuda "$1" --freeze >> "output_main/output_bert_freeze_split_both_${count}.txt" 2>> "output_main/errors_bert_freeze_split_both_${count}.txt"
	done
	
	for lr in 0.0001 0.0003 0.001 0.003
	do
		python3.6 -u lstm_encoder.py --count $count --mode pfx --lexicon shared --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_main/output_lstm_shared_pfx_${count}.txt" 2>> "output_main/errors_lstm_shared_pfx_${count}.txt"
		python3.6 -u lstm_encoder.py --count $count --mode sfx --lexicon shared --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_main/output_lstm_shared_sfx_${count}.txt" 2>> "output_main/errors_lstm_shared_sfx_${count}.txt"
		python3.6 -u lstm_encoder.py --count $count --mode both --lexicon shared --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_main/output_lstm_shared_both_${count}.txt" 2>> "output_main/errors_lstm_shared_both_${count}.txt"
		python3.6 -u lstm_encoder.py --count $count --mode pfx --lexicon split --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_main/output_lstm_split_pfx_${count}.txt" 2>> "output_main/errors_lstm_split_pfx_${count}.txt"
		python3.6 -u lstm_encoder.py --count $count --mode sfx --lexicon split --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_main/output_lstm_split_sfx_${count}.txt" 2>> "output_main/errors_lstm_split_sfx_${count}.txt"
		python3.6 -u lstm_encoder.py --count $count --mode both --lexicon split --batch_size 64 --lr $lr --n_epochs 40 --cuda "$1" >> "output_main/output_lstm_split_both_${count}.txt" 2>> "output_main/errors_lstm_split_both_${count}.txt"
	done
done

