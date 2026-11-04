export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_TwoSIVD

#python -u run.py \
#  --is_training 1 \
#  --root_path ./data/ \
#  --data_path weather.csv \
#  --model_id weather_96_96 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512\
#  --d_ff 512\
#  --Gaussian_dim 256 \
#  --kernel_size 6 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --Gaussian_dim 256 \
#  --kernel_size 3 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./data/ \
#  --data_path weather.csv \
#  --model_id weather_96_192 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512\
#  --d_ff 512\
#  --Gaussian_dim 256 \
#  --kernel_size 6 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --Gaussian_dim 256 \
#  --kernel_size 3 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./data/ \
#  --data_path weather.csv \
#  --model_id weather_96_336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512\
#  --d_ff 512\
#  --Gaussian_dim 256 \
#  --kernel_size 6 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --Gaussian_dim 256 \
#  --kernel_size 6 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./data/ \
#  --data_path weather.csv \
#  --model_id weather_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512\
#  --d_ff 512\
#  --Gaussian_dim 256 \
#  --kernel_size 6 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --Gaussian_dim 256 \
#  --kernel_size 6 \
#  --context_dim 256 \
#  --context_hidden_dim 256 \
#  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --Gaussian_dim 256 \
  --kernel_size 1 \
  --context_dim 336 \
  --context_hidden_dim 336 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --Gaussian_dim 256 \
  --kernel_size 1 \
  --context_dim 336 \
  --context_hidden_dim 336 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --Gaussian_dim 256 \
  --kernel_size 1 \
  --context_dim 256 \
  --context_hidden_dim 256 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --Gaussian_dim 256 \
  --kernel_size 1 \
  --context_dim 256 \
  --context_hidden_dim 256 \
  --itr 1