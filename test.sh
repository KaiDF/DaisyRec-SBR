#!/bin/bash

# 定义要执行的 Python 命令
python_command="python tune.py --model AttenMixer --dataset games --test"

# 定义 seed 值的数组
seeds=(0 10 42 625 2023)

# 循环执行 Python 命令
for seed in "${seeds[@]}"
do
    eval "${python_command} --seed ${seed}"
done