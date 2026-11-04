#python -u run.py \
#    --task_name finetune \
#    --downstream_task classification \
#    --root_path data/Epilepsy/ \
#    --model_id TimeIVD \
#    --model TimeIVD \
#    --data Epilepsy \
#    --e_layers 2 \
#    --d_layers 1 \
#    --input_len 206 \
#    --enc_in 3 \
#    --dec_in 3 \
#    --c_out 3 \
#    --num_classes 4 \
#    --n_heads 16 \
#    --d_model 256 \
#    --Gaussian_dim 256 \
#    --kernel_size 3 \
#    --context_dim 128 \
#    --context_hidden_dim 32 \
#    --d_ff 0 \
#    --patch_len 12 \
#    --stride 12 \
#    --head_dropout 0.1 \
#    --dropout 0.1 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --batch_size 32 \
#    --gpu 0 \
#    --lr_decay 1.0 \
#    --lradj decay \
#    --scheduler cosine \
#    --patience 10 \
#    --learning_rate 0.001 \
#    --pct_start 0.3 \
#    --train_epochs 100

python -u run.py \
    --task_name finetune \
    --downstream_task classification \
    --root_path data/Epilepsy/ \
    --model_id TimeIVD \
    --model TimeIVD \
    --data Epilepsy \
    --e_layers 2 \
    --d_layers 1 \
    --input_len 206 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --num_classes 4 \
    --n_heads 16 \
    --d_model 256 \
    --Gaussian_dim 256 \
    --kernel_size 3 \
    --context_dim 128 \
    --context_hidden_dim 32 \
    --d_ff 0 \
    --patch_len 12 \
    --stride 12 \
    --head_dropout 0.1 \
    --dropout 0.1 \
    --time_steps 1000 \
    --scheduler cosine \
    --batch_size 32 \
    --gpu 0 \
    --lr_decay 1.0 \
    --lradj decay \
    --scheduler cosine \
    --patience 10 \
    --learning_rate 0.002 \
    --pct_start 0.3 \
    --train_epochs 100
