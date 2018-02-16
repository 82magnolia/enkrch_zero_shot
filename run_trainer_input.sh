export CUDA_VISIBLE_DEVICES=0

#baseline 
PROBLEM=translate_zero_shot_exp
MODEL=transformer
HPARAMS=transformer_base_single_gpu

USR_DIR=$PWD
DATA_DIR=/home/magnolia82/data/data
TMP_DIR=$DATA_DIR
TRAIN_DIR=/home/magnolia82/data/data/train_dir

TRAIN_STEPS=10000
EVAL_FREQ=500

t2t-trainer \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --train_steps=$TRAIN_STEPS \ 
    --eval_steps=$EVAL_FREQ \
