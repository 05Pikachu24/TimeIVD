model_name=CASA_ViFilter
seq_len=96
root_path=./
model_id_name=ETTh2

# our method
kernel=3
learning_rate=0.0001

pred_len=96
python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh2.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --kernel $kernel \
  --learning_rate $learning_rate \
  --batch_size 64 \
  --lradj cosine \
  --train_epochs 20 \
  --patience 5 \
  --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
  --split_ratio 0.3 \
  --noise_factor 0.1 \
  --high_freq_emphasis True \
  --itr 1

pred_len=192
python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh2.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --kernel $kernel \
  --learning_rate $learning_rate \
  --batch_size 64 \
  --lradj cosine \
  --train_epochs 20 \
  --patience 5 \
  --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
  --split_ratio 0.5 \
  --noise_factor 0.1 \
  --high_freq_emphasis True \
  --itr 1

pred_len=336
python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh2.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --kernel $kernel \
  --learning_rate $learning_rate \
  --batch_size 64 \
  --lradj cosine \
  --train_epochs 20 \
  --patience 5 \
  --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
  --split_ratio 0.5 \
  --noise_factor 0.1 \
  --high_freq_emphasis True \
  --itr 1

pred_len=720
python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh2.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --kernel $kernel \
  --learning_rate $learning_rate \
  --batch_size 64 \
  --lradj cosine \
  --train_epochs 20 \
  --patience 5 \
  --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
  --split_ratio 0.5 \
  --noise_factor 0.1 \
  --high_freq_emphasis True \
  --itr 1
