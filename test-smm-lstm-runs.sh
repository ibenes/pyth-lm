#!/usr/bin/env bash
EXP_DIR=$1
EXP_NAME=$2
DATA_ROOT=$3
IVEC_EXTRACTOR=$4

python build_output_enhanced_lstm.py \
    --wordlist=$DATA_ROOT/wordlist.txt \
    --ivec-size=50 \
    --unk="<unk>" \
    --emsize=20 \
    --nhid=20 \
    --save=$EXP_DIR/$EXP_NAME.init.lm

# 2b) train and test SMM-LSTM with partial i-vectors
python train-multifile-ivecs.py \
    --train-list=$DATA_ROOT/valid-list.txt \
    --valid-list=$DATA_ROOT/test-list.txt \
    --ivec-extractor=$IVEC_EXTRACTOR \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME.init.lm \
    --save=$EXP_DIR/$EXP_NAME-partial.lm \
    --epochs=1

python eval-multifile-ivecs.py \
    --file-list=$DATA_ROOT/valid-list.txt \
    --ivec-extractor=$IVEC_EXTRACTOR \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME-partial.lm 

# 2c) train and test SMM-LSTM with oracle i-vectors
python train-ivecs-oracle.py \
    --train-list=$DATA_ROOT/valid-list.txt \
    --valid-list=$DATA_ROOT/test-list.txt \
    --test-list=$DATA_ROOT/valid-list.txt \
    --ivec-extractor=$IVEC_EXTRACTOR \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME.init.lm \
    --save=$EXP_DIR/$EXP_NAME-oracle.lm \
    --epochs=1

python eval-ivecs-oracle.py \
    --file-list=$DATA_ROOT/valid-list.txt \
    --ivec-extractor=$IVEC_EXTRACTOR \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME-oracle.lm 

