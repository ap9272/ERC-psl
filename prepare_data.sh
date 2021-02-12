#!/bin/bash

cd Data
python3 data_preprocess.py dailydialog
cd ..
mkdir Original Train Eval

mv Data/*_train.txt Data/*_valid.txt Data/*_test.txt Original/
mv Data/*.txt Train/
mv Train/UttEmo_targets.txt Eval/