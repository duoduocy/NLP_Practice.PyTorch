func_nmt()
{
    NMT_DATA="data/ai_challenger/machine_translation/nmt_t2t_data_all/nmt_all_0303.train.pt"
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py
}

MODEL_TYPE=''
ROOT_DIR=$PWD
echo "run "$MODEL_TYPE
GPU_ID=$2 # Get gpu id
CKPT_PATH=$3 # Get save path
echo "GPU using "$GPU_ID

case "$1" in
     0)MODEL_TYPE='nmt' && func_nmt;;
     *) echo "No input" ;;
esac