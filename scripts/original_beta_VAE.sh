#! /bin/sh

python main.py --dataset celeba --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --beta 10 --env_name trial --display_save_step 10000
