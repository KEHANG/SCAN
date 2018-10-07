#! /bin/sh

python main.py --SCAN --phase beta_VAE --env_name beta_VAE --dataset celeba\
    --seed 7 --lr 1e-4 --batch_size 100 --max_iter 1e6 --beta 53 --z_dim 32
