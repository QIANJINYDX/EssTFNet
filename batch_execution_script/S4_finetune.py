import itertools
import subprocess
from datetime import datetime

# 定义所有需要网格搜索的超参数组合
hyperparam_combinations = itertools.product(
    [1, 2, 4, 8],       # d_model
    [1, 2, 4],       # n_heads
    [1, 2, 3],       # e_layers
    [1, 2, 4, 8],  # d_ff
    [1, 2, 4, 8],  # fnet_d_ff
    [1, 2, 4, 8],   # fnet_d_model
    [1,2,3] # fnet_layers

)

# 固定参数模板
base_command = (
    "python main_single.py "
    "--max_len 1800 "
    "--protein_max_len 600 "
    "--dataset_path 'data/S4.csv' "
    "--feature_path 'data/S4_feature/final/feature4800.csv' "
    "--batch_size 32 "
    "--is_double_stranded False "
    "--is_only_feature False "
    "--us_GNN False "
    "--us_DNASHAPE False "
    "--us_Pretraining False "
    "--pretraining_path 'saved_model/DNABERT-CDHIT-12999500' "
    "--pretraining_name 'DNABERT-2-117M' "
    "--patience 5 "
    "--gpu 1 "
    "--seed 114514 "
    "--is_balance_data False "
    "--is_save_model False "
    "--is_ATFnet True "
    "--result_path 'result/S4_finetune_atf.csv'"
)

# 为每个实验生成唯一时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for combo_idx, (d_model, n_heads, e_layers, d_ff, fnet_d_ff, fnet_d_model, fnet_layers) in enumerate(hyperparam_combinations, 1):
    
    # 构建完整命令
    full_command = (
        f"{base_command} "
        f"--d_model {d_model} "
        f"--n_heads {n_heads} "
        f"--e_layers {e_layers} "
        f"--d_ff {d_ff} "
        f"--fnet_d_ff {fnet_d_ff} "
        f"--fnet_d_model {fnet_d_model} "
        f"--fnet_layers {fnet_layers} "
    )
    
    # 打印进度信息
    print(f"\n{'=' * 50}")
    print(f"Running combination {combo_idx}:")
    print(f"n_heads={n_heads}, e_layers={e_layers}, d_ff={d_ff}")
    print(f"fnet_d_ff={fnet_d_ff}, fnet_d_model={fnet_d_model}")
    print(f"{'=' * 50}\n")
    
    # 执行命令并处理异常
    try:
        subprocess.run(full_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred with combination {combo_idx}: {str(e)}")
        # 可以选择记录失败案例到日志文件
        with open("failed_experiments.log", "a") as f:
            f.write(f"{datetime.now()} - Failed combo: {combo_idx}\n{full_command}\n\n")