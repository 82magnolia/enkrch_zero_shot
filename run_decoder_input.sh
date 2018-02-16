export CUDA_VISIBLE_DEVICES=0

#baseline 
PROBLEM=translate_zero_shot_exp
MODEL=transformer
HPARAMS=transformer_base_single_gpu
BEAM_SIZE=4
ALPHA=0.6
USR_DIR=$PWD
DATA_DIR=/home/magnolia82/data/data
TMP_DIR=$DATA_DIR
TRAIN_DIR=/home/magnolia82/data/data/train_dir
DECODE_FILE=$DATA_DIR/dev/chinese.txt #TODO:Change file


TRAIN_STEPS=10000
EVAL_FREQ=500

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=chkr_translation.txt \
ã
