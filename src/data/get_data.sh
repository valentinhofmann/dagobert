#!/bin/bash

for file in ../../data/raw/*
do
  python3.6 -u get_sents.py $file &
done

python3.6 -u filter_sents.py

python3.6 -u get_final.py 0.6 0.2

python3.6 -u get_final_split.py 0.6 0.2
