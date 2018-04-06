#!/usr/bin/env bash
EXP_DIR=$1

WT_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos/
IVEC_EXTRACTOR=/mnt/matylda5/ibenes/projects/santosh-lm/smm-models/wt2-model/extractor-k50

# note that only validation part of WT-2 is used for training to speed the test up

# 1) build a standard LSTM
./test-lstm-runs.sh $EXP_DIR lsmt $WT_ROOT

# 2) build a SMM-LSTM
./test-smm-lstm-runs.sh $EXP_DIR smm-lstm $WT_ROOT $IVEC_EXTRACTOR

# 3) build iFN-LM
./test-fn-runs.sh $EXP_DIR ifn $WT_ROOT 

# 4) build iFN-LM
./test-ifn-runs.sh $EXP_DIR ifn $WT_ROOT $IVEC_EXTRACTOR
