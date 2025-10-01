#!/usr/bin/bash
#SBATCH -J ModernTCN
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id ETTh1_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 7 \
  --dropout 0.3 \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_ETTh1_336_336.log

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id ETTh2_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 7 \
  --dropout 0.3 \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_ETTh2_336_336.log

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id ETTm1_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 3 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.1 \
  --enc_in 7 \
  --dropout 0.3 \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_ETTm1_336_336.log

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id ETTm2_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 3 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.2 \
  --enc_in 7 \
  --dropout 0.8 \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_ETTm2_336_336.log

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id Exchange_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 1 \
  --patch_size 1 \
  --patch_stride 1 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.6 \
  --enc_in 8 \
  --dropout 0.2 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_exchange_336_336.log

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id traffic_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path traffic.csv \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 862 \
  --dropout 0.9 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_traffic_336_336.log

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id weather_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path weather.csv \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 21 \
  --dropout 0.4 \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_weather_336_336.log

torchrun --nproc_per_node=2 --master-port 12346 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --model_id electricity_336_336 \
  --model ModernTCN \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path electricity.csv \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 321 \
  --dropout 0.9 \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/ModernTCN_electricity_336_336.log
