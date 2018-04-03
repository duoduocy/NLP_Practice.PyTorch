#!/usr/bin/env bash
clear
ROOT_DIR=$PWD
echo "Set root dir to: ""$ROOT_DIR"

func_nmt_eval()
{
    SOURCE_DIR=$PWD
    eval cd "${SOURCE_DIR}"
    export PYTHONPATH="$PYTHONPATH:$SOURCE_DIR"
    echo "Start from source dir: "$SOURCE_DIR
    MODEL_NAME=$SOURCE_DIR/demo-model_acc_60.34_ppl_6.72_e13.pt
    SRC_FILE=$SOURCE_DIR/data/aic_mt/nmt_t2t_data_all/valid_0303.zh
    TGT_FILE=$SOURCE_DIR/tmp/aic_mt_val.en.txt
    echo "Model:"$MODEL_NAME "Src:"$SRC_FILE "Tgt:"$TGT_FILE
    python translate.py -model $MODEL_NAME -src $SRC_FILE -output $TGT_FILE -verbose -gpu 0
}

func_nmt_eval