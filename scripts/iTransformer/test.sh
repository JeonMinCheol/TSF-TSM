#!/usr/bin/bash
#SBATCH -J iTransformer
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
model_name=iTransformer

root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
d_model=128
d_ff=128
features=M
e_layers=2
n_heads=8
random_seed=2021

data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
for pred_len in 336
do
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
      --d_ff $d_ff \
      --d_model $d_model \
      --des 'Exp' \
      --train_epochs 200\
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

for pred_len in 336
do
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
      --d_ff $d_ff \
      --d_model $d_model \
      --des 'Exp' \
      --train_epochs 200\
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

for pred_len in 336
do
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
      --des 'Exp' \
      --train_epochs 200\
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

for pred_len in 336
do
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
      --d_ff $d_ff \
      --d_model $d_model \
      --des 'Exp' \
      --train_epochs 200\
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom

for pred_len in 336
do
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
      --e_layers 2 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --n_heads $n_heads \
      --des 'Exp' \
      --train_epochs 200\
      --d_model 128 \
      --d_ff 128 \
      --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

data_path_name=weather.csv
model_id_name=weather
data_name=custom

for pred_len in 336
do
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
      --dec_in 21 \
      --c_out 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 512\
      --d_ff 512\
      --des 'Exp' \
      --train_epochs 200\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'336 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --d_model 512\
  --d_ff 512 \
  --enc_in 321 \
  --des 'Exp' \
  --train_epochs 200 \
  --itr 1 --batch_size 16  --learning_rate 0.0005  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  

data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'336 \
  --model $model_name \
  --data $data_name \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --features M \
  --seq_len $seq_len \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --pred_len 336 \
  --itr 1 > logs/LongForecasting/$model_name'_'traffic_$seq_len'_'336.log  
