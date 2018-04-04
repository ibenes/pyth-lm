#!/usr/bin/env bash
EXP_DIR=$1

WT_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos/
IVEC_EXTRACTOR=/mnt/matylda5/ibenes/projects/santosh-lm/smm-models/wt2-model/extractor-k50

# note that only validation part of WT-2 is used for training to speed the test up

# 1a) build a standard LSTM
LSTM_NAME=lstm
./test-lstm-runs.sh $EXP_DIR $LSTM_NAME $WT_ROOT

# 2a) build a SMM-LSTM
SMM_LSTM_NAME=smm-lstm
python build_output_enhanced_lstm.py --wordlist=$WT_ROOT/wordlist.txt --ivec-size=50 --unk="<unk>" --emsize=20 --nhid=20 --save=$EXP_DIR/$SMM_LSTM_NAME.init.lm

# 2b) train and test SMM-LSTM with partial i-vectors
python train-multifile-ivecs.py --train-list=$WT_ROOT/valid-list.txt --valid-list=$WT_ROOT/test-list.txt --test-list=$WT_ROOT/valid-list.txt --ivec-extractor=$IVEC_EXTRACTOR --concat-articles --cuda --load=$EXP_DIR/$SMM_LSTM_NAME.init.lm --save=$EXP_DIR/$SMM_LSTM_NAME-partial.lm --epochs=1

python eval-multifile-ivecs.py --file-list=$WT_ROOT/valid-list.txt --ivec-extractor=$IVEC_EXTRACTOR --concat-articles --cuda --load=$EXP_DIR/$SMM_LSTM_NAME-partial.lm 

# 2c) train and test SMM-LSTM with oracle i-vectors
python train-ivecs-oracle --train-list=$WT_ROOT/valid-list.txt --valid-list=$WT_ROOT/test-list.txt --test-list=$WT_ROOT/valid-list.txt --ivec-extractor=$IVEC_EXTRACTOR --concat-articles --cuda --load=$EXP_DIR/$SMM_LSTM_NAME.init.lm --save=$EXP_DIR/$SMM_LSTM_NAME-oracle.lm --epochs=1

python eval-ivecs-oracle.py --file-list=$WT_ROOT/valid-list.txt --ivec-extractor=$IVEC_EXTRACTOR --concat-articles --cuda --load=$EXP_DIR/$SMM_LSTM_NAME-oracle.lm 

# 3a) build iFN-LM
IFN_NAME=ifn-lm
python build_bengio_ivec_input.py --wordlist=$WT_ROOT/wordlist.txt --ivec-dim=50 --unk="<unk>" --emsize=20 --nhid=20 --save=$EXP_DIR/$IFN_NAME.init.lm

# 3b) train the iFN-LM with oracle ivectors and evaluate using partial ones
python train-ivecs-oracle --train-list=$WT_ROOT/valid-list.txt --valid-list=$WT_ROOT/test-list.txt --test-list=$WT_ROOT/valid-list.txt --ivec-extractor=$IVEC_EXTRACTOR --concat-articles --cuda --load=$EXP_DIR/$IFN_NAME.init.lm --save=$EXP_DIR/$IFN_NAME.lm --epochs=1

python eval-multifile-ivecs.py --file-list=$WT_ROOT/valid-list.txt --ivec-extractor=$IVEC_EXTRACTOR --concat-articles --cuda --load=$EXP_DIR/$IFN_NAME.lm 
