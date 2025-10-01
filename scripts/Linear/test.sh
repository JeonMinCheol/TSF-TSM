#!/usr/bin/bash
#SBATCH -J DLinear
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v1
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear
root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
d_model=256
d_ff=256
n_heads=8
features=M
e_layers=3
random_seed=2021
pred_len=336

# data_path_name=ETTh1.csv
# model_id_name=ETTh1
# data_name=ETTh1

# torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features $features \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers $e_layers \
#       --n_heads $n_heads \
#       --d_ff $d_ff \
#       --d_model $d_model \
#       --hidden_features 256 \
#       --flow_layers 8 \
#       --num_bins 12 \
#       --des 'Exp' \
#       --alpha 1.0\
#       --moving_avg 24\
#       --stride 12\
#       --momentum 0.99\
#       --num_experts 6\
#       --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=ETTh2.csv
# model_id_name=ETTh2
# data_name=ETTh2

# torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features $features \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers $e_layers \
#       --n_heads $n_heads \
#       --d_ff $d_ff \
#       --d_model $d_model \
#       --hidden_features 256 \
#       --flow_layers 8 \
#       --num_bins 12 \
#       --des 'Exp' \
#       --alpha 1.0\
#       --moving_avg 24\
#       --stride 12\
#       --momentum 0.99\
#       --num_experts 6\
#       --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
# data_path_name=ETTm1.csv
# model_id_name=ETTm1
# data_name=ETTm1

# torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features $features \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers $e_layers \
#       --n_heads $n_heads \
#       --d_ff $d_ff \
#       --d_model $d_model \
#       --hidden_features 256 \
#       --flow_layers 8 \
#       --num_bins 12 \
#       --des 'Exp' \
#       --alpha 1.0\
#       --moving_avg 24\
#       --stride 12\
#       --momentum 0.99\
#       --num_experts 6\
#       --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
# data_path_name=ETTm2.csv
# model_id_name=ETTm2
# data_name=ETTm2

# torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features $features \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers $e_layers \
#       --n_heads $n_heads \
#       --d_ff $d_ff \
#       --d_model $d_model \
#       --hidden_features 256 \
#       --flow_layers 8 \
#       --num_bins 12 \
#       --des 'Exp' \
#       --alpha 1.0\
#       --moving_avg 24\
#       --stride 12\
#       --momentum 0.99\
#       --num_experts 6\
#       --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features $features \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --des 'Exp' \
      --alpha 1.0\
      --stride 12\
      --momentum 0.99\
      --patience 25 \
      --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --patience 25 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name_$seq_len'_'336 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --patience 25 \
  --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 862 \
  --des 'Exp' \
  --patience 25 \
  --itr 1 --batch_size 16 --learning_rate 0.05   >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'336.log  
