func_nmt()
{
    NMT_DATA="data/ai_challenger/machine_translation/nmt_t2t_data_all/nmt_all_0303.train.pt"
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_nmt.py -data $NMT_DATA -save_model demo-model-0303-full  -batch_size 128 -gpus 0 -loss_type 2 -dropout 0.5 -optim adam -learning_rate 1e-3 -train_from save/nmt/demo-model-0303/demo-model-0303-full_acc_54.88_ppl_8.93_e13.pt
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