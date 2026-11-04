export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_TwoSIVD

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --Gaussian_dim 256 \
  --kernel_size 6 \
  --context_dim 256 \
  --context_hidden_dim 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --Gaussian_dim 256 \
  --kernel_size 6 \
  --context_dim 256 \
  --context_hidden_dim 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --Gaussian_dim 256 \
  --kernel_size 6 \
  --context_dim 336 \
  --context_hidden_dim 336 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --Gaussian_dim 256 \
  --kernel_size 6 \
  --context_dim 336 \
  --context_hidden_dim 336 \
  --itr 1