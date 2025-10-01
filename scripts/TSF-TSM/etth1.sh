#!/usr/bin/bash
#SBATCH -J env_model_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v7
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

OMP_NUM_THREADS=16

# add --individual for TSF_TSM-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=TSF_TSM

root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=24
for pred_len in 96 #192 336 720
do
torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 8 \
      --d_ff 256 \
      --d_model 256 \
      --hidden_features 128 \
      --flow_layers 5 \
      --num_bins 12 \
      --des 'Exp' \
      --train_epochs 100\
      --alpha 1\
      --training_stage 1 \
      --itr 1 --batch_size 1024 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 8 \
      --d_ff 256 \
      --d_model 256 \
      --hidden_features 128 \
      --flow_layers 5 \
      --num_bins 12 \
      --des 'Exp' \
      --train_epochs 100\
      --alpha 1\
      --training_stage 2 \
      --stage1_path './checkpoints/'$seq_len'_'$pred_len'_'$model_name'_'$data_name'_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth' \
      --itr 1 --batch_size 1024 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
