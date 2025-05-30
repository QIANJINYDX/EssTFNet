#!/bin/bash

# 定义参数范围
n_heads_values=(2 4 8 16)
e_layers_values=(1 2 3)
d_ff_values=(256 512 1024)

# 使用嵌套循环遍历所有参数组合
for n_heads in "${n_heads_values[@]}"; do
  for e_layers in "${e_layers_values[@]}"; do
    for d_ff in "${d_ff_values[@]}"; do
      # 执行 Python 命令
      echo "Running with n_heads=$n_heads, e_layers=$e_layers, d_ff=$d_ff"
      python main.py --n_heads $n_heads --e_layers $e_layers --d_ff $d_ff --batch_size 256 --is_double_stranded False --is_only_feature False --us_GNN False --us_DNASHAPE False --us_Pretraining False\
      --patience 5
    done
  done
done
