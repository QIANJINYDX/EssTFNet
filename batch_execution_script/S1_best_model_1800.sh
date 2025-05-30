#!/bin/bash
# S1数据集确定最优特征数量
for i in {50..8250..50}; do
  python main.py --max_len 1800 \
  --dataset_path "data/S1_new.csv" \
  --feature_path "data/S1_feature/final/feature$i.csv" \
  --batch_size 64 \
  --is_double_stranded False \
  --is_only_feature False \
  --us_GNN False \
  --us_DNASHAPE False \
  --is_ATFnet False \
  --is_motif_extraction False \
  --patience 5 \
  --gpu 3 \
  --result_path "result/S1_1800bp.csv" \
  --seed 114514 \
  --is_balance_data False
done
 
