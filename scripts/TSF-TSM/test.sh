#!/usr/bin/bash
#SBATCH -J TSF_TSM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-g1
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

OMP_NUM_THREADS=16

root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
seq_len=336
model_name=TSF_TSM
d_model=256
d_ff=256
n_heads=8
features=M
e_layers=4
hidden_features=256
pred_len=336
random_seed=2021
flow_layers=8
num_bins=12

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
#       --c_out 7 \
#       --e_layers $e_layers \
#       --n_heads $n_heads \
#       --d_ff $d_ff \
#       --d_model $d_model \
#       --hidden_features $hidden_features \
#       --flow_layers $flow_layers \
#       --num_bins $num_bins \
#       --des 'Exp' \
#       --moving_avg 60\
#       --stride 10\
#       --alpha 0.5\
#       --momentum 0.9\
#       --num_experts 6 \
#       --dropout 0.2 \
#       --use_multi_gpu \
#       --devices 0,1,2,3 \
#       --itr 1 --batch_size 512 --learning_rate 0.0003  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

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
#       --c_out 7 \
#       --e_layers $e_layers \
#       --n_heads $n_heads \
#       --d_ff $d_ff \
#       --d_model $d_model \
#       --hidden_features $hidden_features \
#       --flow_layers $flow_layers \
#       --num_bins $num_bins \
#       --des 'Exp' \
#       --alpha 0.5\
#       --moving_avg 60\
#       --stride 10\
#       --momentum 0.99\
#       --dropout 0.3\
#       --num_experts 6 \
#       --use_multi_gpu \
#       --devices 0,1,2,3 \
#       --itr 1 --batch_size 512 --learning_rate 0.0003  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

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
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --hidden_features $hidden_features \
      --flow_layers $flow_layers \
      --num_bins $num_bins \
      --des 'Exp' \
      --alpha 0.1\
      --moving_avg 60\
      --stride 10\
      --momentum 0.99\
      --num_experts 6\
      --dropout 0.3 \
      --devices 0,1,2,3 \
      --itr 1 --batch_size 512 --learning_rate 0.0005  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

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
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --hidden_features $hidden_features \
      --flow_layers $flow_layers \
      --num_bins $num_bins \
      --des 'Exp' \
      --alpha 1.0\
      --moving_avg 60\
      --stride 10\
      --momentum 0.99\
      --num_experts 6\
      --dropout 0.3 \
      --devices 0,1,2,3 \
      --itr 1 --batch_size 512 --learning_rate 0.0005  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=exchange_rate.csv
# model_id_name=Exchange
# data_name=custom

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
#       --enc_in 8 \
#       --c_out 8 \
#       --e_layers $e_layers \
#       --n_heads $n_heads \
#       --d_ff $d_ff \
#       --d_model $d_model \
#       --hidden_features $hidden_features \
#       --flow_layers $flow_layers \
#       --num_bins $num_bins \
#       --des 'Exp' \
#       --alpha 0.01\
#       --moving_avg 60\
#       --stride 10\
#       --momentum 0.5\
#       --num_experts 4\
#       --dropout 0.6 \
#       --devices 0,1,2,3 \
#       --itr 1 --batch_size 8 --learning_rate 0.0005  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=weather.csv
# model_id_name=weather
# data_name=custom

# d_model=128
# d_ff=128
# n_heads=16
# features=M
# e_layers=3
# hidden_features=128
# pred_len=336
# random_seed=2021
# flow_layers=4
# num_bins=8

# torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 21 \
#       --e_layers 3 \
#       --n_heads 16 \
#       --d_model 128 \
#       --d_ff 128 \
#       --dropout 0.2\
#       --moving_avg 24\
#       --stride 12\
#       --des 'Exp' \
#       --alpha 0.01\
#       --devices 0,1,2,3 \
#       --itr 1 --batch_size 512 --learning_rate 0.0001  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 


data_path_name=electricity.csv
model_id_name=electricity
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
      --enc_in 321 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --dropout 0.6 \
      --hidden_features $hidden_features \
      --flow_layers $flow_layers \
      --num_bins $num_bins \
      --des 'Exp' \
      --alpha 0.001\
      --moving_avg 60\
      --stride 10\
      --momentum 0.9\
      --num_experts 4 \
      --use_multi_gpu \
      --devices 0,1,2,3 \
      --itr 1 --batch_size 48 --learning_rate 0.0005 > logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  


data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

d_model=64
d_ff=64
hidden_features=64
flow_layers=5
num_bins=6

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
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --hidden_features $hidden_features \
      --enc_in 862 \
      --flow_layers $flow_layers \
      --num_bins $num_bins \
      --des 'Exp' \
      --alpha 1.0\
      --moving_avg 60\
      --stride 10\
      --dropout 0.3 \
      --momentum 0.9\
      --num_experts 6\
      --use_multi_gpu \
      --devices 0,1,2,3 \
      --itr 1 --batch_size 16 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'traffic_$seq_len'_'336.log  