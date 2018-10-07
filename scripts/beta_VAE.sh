#! /bin/sh

python main.py --SCAN --phase DAE --dataset celeba --seed 1 --lr 1e-3 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --beta 10 --env_name trial --display_save_step 10000
