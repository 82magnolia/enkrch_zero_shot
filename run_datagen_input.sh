PROBLEM=translate_zero_shot_exp
MODEL=transformer
HPARAMS=transformer_base_single_gpu

USR_DIR=$PWD
DATA_DIR=/home/magnolia82/data/data
TMP_DIR=$DATA_DIR
TRAIN_DIR=/home/magnolia82/data/data/train_dir

t2t-datagen \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM
