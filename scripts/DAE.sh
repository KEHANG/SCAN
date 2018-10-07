#! /bin/sh

python main.py --SCAN --phase DAE --env_name DAE --dataset celeba\ 
    --seed 3 --lr 1e-3 --batch_size 100 --max_iter 2e5 --z_dim 100
