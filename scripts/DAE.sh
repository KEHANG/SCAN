#! /bin/sh

python main.py --SCAN --phase DAE --dataset celeba --seed 3 --lr 1e-3 --batch_size 100 --max_iter 2e5 --env_name DAE\
    --display_save_step 1000
