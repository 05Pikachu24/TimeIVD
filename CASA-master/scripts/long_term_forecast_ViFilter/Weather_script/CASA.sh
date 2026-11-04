model_name=CASA_ViFilter
seq_len=96
root_path=./
model_id_name=weather

# our method
kernel=3
learning_rate=0.0003


for pred_len in 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path weather.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model 512 \
    --kernel $kernel \
    --learning_rate $learning_rate \
    --batch_size 32 \
    --lradj cosine \
    --train_epochs 10 \
    --patience 10 \
    --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
    --split_ratio 0.5 \
    --noise_factor 0.05 \
    --high_freq_emphasis False \
    --itr 1
done

for pred_len in 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path weather.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model 512 \
    --kernel $kernel \
    --learning_rate $learning_rate \
    --batch_size 32 \
    --lradj cosine \
    --train_epochs 10 \
    --patience 10 \
    --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
    --split_ratio 0.3 \
    --noise_factor 0.1 \
    --high_freq_emphasis True \
    --itr 1
done



