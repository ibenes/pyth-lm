#!/usr/bin/env bash
EXP_DIR=$1
EXP_NAME=$2
DATA_ROOT=$3
IVEC_EXTRACTOR=$4

python build_bengio.py \
    --wordlist=$DATA_ROOT/wordlist.txt \
    --unk="<unk>" \
    --emsize=20 \
    --nhid=20 \
    --save=$EXP_DIR/$EXP_NAME.init.lm

# 1) train the FN-LM with oracle ivectors and evaluate using partial ones
python train-ff-multifile.py \
    --train-list=$DATA_ROOT/valid-list.txt \
    --valid-list=$DATA_ROOT/test-list.txt \
    --test-list=$DATA_ROOT/valid-list.txt \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME.init.lm \
    --save=$EXP_DIR/$EXP_NAME.lm \
    --epochs=1

python eval-ff-multifile.py \
    --file-list=$DATA_ROOT/valid-list.txt \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME.lm 
