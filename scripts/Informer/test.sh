#!/usr/bin/bash
#SBATCH -J Informer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v1
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

random_seed=2021
model_name=Informer

for pred_len in 336
do

#   torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --is_training 1 \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --data_path ETTh1.csv \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --model_id ETTh1_336_$pred_len \
#       --model $model_name \
#       --data ETTh1 \
#       --features M \
#       --seq_len 336 \
#       --label_len 48 \
#       --pred_len $pred_len \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --des 'Exp' \
#       --itr 1 >logs/LongForecasting/$model_name'_Etth1_'$pred_len.log
  
#   torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --is_training 1 \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --data_path ETTh2.csv \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --model_id ETTh2_336_$pred_len \
#       --model $model_name \
#       --data ETTh2 \
#       --features M \
#       --seq_len 336 \
#       --label_len 48 \
#       --pred_len $pred_len \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --des 'Exp' \
#       --itr 1 >logs/LongForecasting/$model_name'_Etth2_'$pred_len.log
  
#   torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --is_training 1 \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --data_path ETTm1.csv \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --model_id ETTm1_336_$pred_len \
#       --model $model_name \
#       --data ETTm1 \
#       --features M \
#       --seq_len 336 \
#       --label_len 48 \
#       --pred_len $pred_len \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --des 'Exp' \
#       --itr 1 >logs/LongForecasting/$model_name'_Ettm1_'$pred_len.log

#   torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#       --is_training 1 \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --data_path ETTm2.csv \
#       --model_id ETTm2_336_$pred_len \
#       --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#       --model $model_name \
#       --data ETTm2 \
#       --features M \
#       --seq_len 336 \
#       --label_len 48 \
#       --pred_len $pred_len \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --des 'Exp' \
#       --itr 1 >logs/LongForecasting/$model_name'_Ettm2_'$pred_len.log

#   torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#     --is_training 1 \
#     --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#     --data_path exchange_rate.csv \
#     --model_id exchange_336_$pred_len \
#     --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 336 \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'Exp' \
#     --itr 1 >logs/LongForecasting/$model_name'_exchange_rate_'$pred_len.log

  torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
      --is_training 1 \
      --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
      --data_path electricity.csv \
      --model_id electricity_336_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 336 \
      --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 >logs/LongForecasting/$model_name'_electricity_'$pred_len.log

#   torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#     --is_training 1 \
#     --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#     --data_path traffic.csv \
#     --model_id traffic_336_$pred_len \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 336 \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --itr 1  >logs/LongForecasting/$model_name'_traffic_'$pred_len.log

#   torchrun --nproc_per_node=1 /data/a2019102224/PatchTST_supervised/run_longExp.py \
#     --is_training 1 \
#     --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#     --data_path weather.csv \
#     --model_id weather_336_$pred_len \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 336 \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
#     --c_out 21 \
#     --des 'Exp' \
#     --itr 1  >logs/LongForecasting/$model_name'_weather_'$pred_len.log
done

