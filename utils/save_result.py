import csv
import time
import torchvision

def write_result(feature_len, emb_type, args, gene_name, metrics, result_path):
    """
    保存结果（包含分类和回归指标）
    :param feature_len: 特征长度
    :param emb_type: 嵌入类型
    :param args: 参数
    :param gene_name: 基因名称
    :param metrics: 分类指标
    :param result_path: 结果路径
    :return:
    """
    # 定义CSV文件的列名
    columns = [
        'gene_name', 'feature_len', 'emb_type', 'seed', 'max_len', 'kernel_size', 
        'head_num', 'hidden_dim', 'layer_num', 'attn_drop', 'lstm_drop', 'linear_drop',
        'lr', 'wd', 'batch_size', 'epoch_num', 'patience', 'threshold',
        'is_double_stranded', 'is_only_feature', 'us_GNN', 'us_DNASHAPE',
        'us_Pretraining', 'is_motif_extraction', 'is_ATFnet', 'us_protein',
        # 分类指标
        'tp', 'tn', 'fp', 'fn', 'acc', 'f1', 'pre', 'rec', 'mcc', 'AUC', 'AUPR',
        # 区分指标
        "note","pretraining_path",
        # ATFNet参数
        "d_model","factor","n_heads","e_layers","d_ff","atf_drop","fnet_d_ff","fnet_d_model","complex_dropout","fnet_layers","is_emb"

    ]
    
    # 准备数据行
    tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = metrics
    row = [
        gene_name, feature_len, emb_type, args["seed"], args["max_len"], args["kernel_size"],
        args["head_num"], args["hidden_dim"], args["layer_num"], args["attn_drop"], args["lstm_drop"], args["linear_drop"],
        args["lr"], args["wd"], args["batch_size"], args["epoch_num"], args["patience"], args["threshold"],
        args["is_double_stranded"], args["is_only_feature"], args["us_GNN"], args["us_DNASHAPE"],
        args["us_Pretraining"], args["is_motif_extraction"], args["is_ATFnet"], args["us_protein"],
        # 分类指标
        tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR,
        # 区分指标
        args["note"],args["pretraining_path"],
        # ATFNet参数
        args["d_model"],args["factor"],args["n_heads"],args["e_layers"],args["d_ff"],args["atf_drop"],args["fnet_d_ff"],args["fnet_d_model"],args["complex_dropout"],args["fnet_layers"],args["is_emb"]
    ]
    
    # 检查文件是否存在
    file_exists = False
    try:
        with open(result_path, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    # 写入数据
    with open(result_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if not file_exists:
            writer.writerow(columns)
        writer.writerow(row)

def write_regression_result(feature_len, emb_type, args, gene_name, regression_metrics, result_path):
    """
    保存回归任务的结果
    :param feature_len: 特征长度
    :param emb_type: 嵌入类型
    :param args: 参数
    :param gene_name: 基因名称
    :param regression_metrics: 回归指标
    :param result_path: 结果路径
    :return:
    """
    # 定义CSV文件的列名
    columns = [
        'gene_name', 'feature_len', 'emb_type', 'seed', 'max_len', 'kernel_size', 
        'head_num', 'hidden_dim', 'layer_num', 'attn_drop', 'lstm_drop', 'linear_drop',
        'lr', 'wd', 'batch_size', 'epoch_num', 'patience', 'threshold',
        'is_double_stranded', 'is_only_feature', 'us_GNN', 'us_DNASHAPE',
        'us_Pretraining', 'is_motif_extraction', 'is_ATFnet', 'us_protein',
        'MSE', 'RMSE', 'MAE', 'R2', 'Adjusted_R2','r'
    ]
    
    # 准备数据行
    row = [
        gene_name, feature_len, emb_type, args["seed"], args["max_len"], args["kernel_size"],
        args["head_num"], args["hidden_dim"], args["layer_num"], args["attn_drop"], args["lstm_drop"], args["linear_drop"],
        args["lr"], args["wd"], args["batch_size"], args["epoch_num"], args["patience"], args["threshold"],
        args["is_double_stranded"], args["is_only_feature"], args["us_GNN"], args["us_DNASHAPE"],
        args["us_Pretraining"], args["is_motif_extraction"], args["is_ATFnet"], args["us_protein"],
        regression_metrics['MSE'], regression_metrics['RMSE'], regression_metrics['MAE'],
        regression_metrics['R2'], regression_metrics['Adjusted_R2'],regression_metrics['r']
    ]
    
    # 将结果路径的扩展名改为_regression.csv
    regression_result_path = result_path.replace('.txt', '_regression.csv')
    
    # 检查文件是否存在
    file_exists = False
    try:
        with open(regression_result_path, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    # 写入数据
    with open(regression_result_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if not file_exists:
            writer.writerow(columns)
        writer.writerow(row)

def save_model_structures(model,gene_name,save_path):
    file_exists = False
    try:
        with open(save_path, mode='r') as file:
            file_exists = True
    except FileNotFoundError:
        pass
    with open(save_path, mode='a', newline='') as file:
        wtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        file.write(str(gene_name)+" "+str(wtime)+"\n")

        for i in model.children():
            file.write(str(i))
        file.write("\n"+"-------------------------------------\n")

# resnet=torchvision.models.resnet18(pretrained=True)
# save_model(resnet,"23132-87","result/Model_structure.txt")