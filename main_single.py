import warnings, os, random, torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from module import DSEG, FocalLoss, initial_model
from utils import load_dataset, set_params, cal_metrics, print_metrics, best_acc_thr
from utils.cal_metrics import cal_regression_metrics, print_regression_metrics
import torch.nn as nn
from loguru import logger
from utils.save_result import write_result, write_regression_result
from utils.load_bigmodel import BigPreTrainModel
import pandas as pd
from utils import ATFNetConfigs
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
args = set_params()

logger.add(args.log_path, rotation="500 MB", compression="zip", serialize=True)
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHUSHSEED'] = str(random_seed)

warnings.filterwarnings('ignore')
if torch.cuda.is_available():
    num_workers = 4
    device = torch.device(f"cuda")
    torch.cuda.set_device(args.gpu)  # 默认设备为 0
    # device_ids = [0, 2]
else:
    num_workers = 0
    device = torch.device("cpu")
# num_workers = 0
# device = torch.device("cpu")


def log_info(info_dict):
    logger.info("Hyperparameters:")
    for key, value in info_dict.items():
        logger.info(f"{key}: {value}")



def train_cv(fitness=None, DSEG_label=None, gene_name="None", result_path=None, model_structures_path="result/Model_structures.txt", save_path=None):
    """
    训练模型
    :param DSEG_label: label
    :param gene_name: 基因名称
    :param result_path: 结果路径
    :param model_structures_path: 保存模型结构路径
    :param save_path: 保存模型路径
    :return:
    """
    # 导入超参数
    seed, seq_type, emb_type, max_len, epoch_num, batch_size, patience, threshold = \
        args.seed, args.seq_type, args.emb_type, args.max_len, args.epoch_num, args.batch_size, args.patience, args.threshold

    max_len, kernel_size, head_num, hidden_dim, layer_num, attn_drop, lstm_drop, linear_drop = \
        args.max_len, args.kernel_size, args.head_num, args.hidden_dim, args.layer_num, args.attn_drop, args.lstm_drop, args.linear_drop
    
    # 确定任务类型
    if fitness is not None:
        task_type = 'regression'
    else:
        task_type = 'classification'
    
    log_info(args.__dict__)
    save_path = args.save_path
    logger.info(f'==============New Training==============')
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"MaxLen:{max_len}")
    logger.info(f"Task Type: {task_type}")
    
    # Load datasets
    if args.us_Pretraining==True and args.pretraining_path!=None:
        big_model=BigPreTrainModel(args.pretraining_path,args.pretraining_name,device)
        big_model.load_model()
        big_model.load_tokenizer()
    else:
        big_model=None
        
    train_dataset, test_dataset, F_len = load_dataset(seq_type, emb_type, max_len, args.protein_max_len, args.feature_path, seed, args.dataset_path, args.dnashape_path, args.pretraining_path, DSEG_label=DSEG_label,
                                                      us_GNN=args.us_GNN, us_DNASHAPE=args.us_DNASHAPE,
                                                      us_Pretraining=args.us_Pretraining, is_balance_data=args.is_balance_data,
                                                      big_model=big_model,
                                                      is_motif_extraction=args.is_motif_extraction,
                                                      fitness=fitness,
                                                      protein_feature_path=args.protein_feature_path)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
    )
    
    logger.info(f"FeatureLen:{F_len}")
    # atfconfig
    atfconfig=ATFNetConfigs(max_len, train_dataset.emb_dim, max_len, args.d_model, args.factor, args.n_heads, args.e_layers, args.d_ff, args.atf_drop, args.fnet_d_ff, args.fnet_d_model, args.complex_dropout, args.fnet_layers, args.is_emb)
    # Model 14 值DNASHAPE是14维
    shape_atf_config=ATFNetConfigs(max_len, 14, max_len, args.d_model, args.factor, args.n_heads, args.e_layers, args.d_ff, args.atf_drop, args.fnet_d_ff, args.fnet_d_model, args.complex_dropout, args.fnet_layers, args.is_emb)
    
    model = DSEG(max_len, 
                train_dataset.emb_dim, 
                args.protein_max_len,
                train_dataset.protein_emb_dim,
                kernel_size, 
                head_num, hidden_dim, layer_num, 
                attn_drop, lstm_drop, linear_drop, atfconfig, shape_atf_config,
                F_len,
                args.is_double_stranded,
                args.is_only_feature,
                us_gnn=args.us_GNN,
                us_DNASHAPE=args.us_DNASHAPE,
                us_Pretraining=args.us_Pretraining,
                pretraining_path=args.pretraining_path,
                is_motif_extraction=args.is_motif_extraction,
                is_ATFnet=args.is_ATFnet,
                big_model=big_model,
                us_protein=args.us_protein,
                is_esm=args.is_esm,
                task_type=task_type,
                us_protein_table=args.us_protein_table)  # 添加任务类型参数
    
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.wd)
    
    # Loss functions
    if task_type == 'classification':
        # criterion = nn.BCELoss()
        # criterion = FocalLoss(alpha=0.25, gamma=2)
        # Loss function
        pos_weight = float(train_dataset.num_non / train_dataset.num_ess)
        criterion = FocalLoss(gamma=0, pos_weight=pos_weight, logits=False, reduction='sum')
    else:  # regression
        criterion = nn.MSELoss()
    
    # Train and validation using 5-fold cross validation
    val_metrics_list, test_metrics_list = [], []  # 用于存储每个fold的指标
    kfold_test_scores = []
    kfold = 5
    all_best_model = []  # 保存所有的最优模型
    final_best_model=None
    skf = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True)
    regression_best={}
    max_auc=0
    
    for i, (train_index, val_index) in enumerate(skf.split(train_dataset.features, train_dataset.labels)):
        logger.info(f'\nStart training CV fold {i + 1}:')
        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False,
                                  num_workers=num_workers, worker_init_fn=np.random.seed(seed))
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False,
                                num_workers=num_workers, worker_init_fn=np.random.seed(seed))

        # Train model
        initial_model(model)

        count = 0
        if task_type != 'classification':
            best_val_metric=-10
        else:
            best_val_metric = 0
        best_test_metric = 0
        best_test_scores = []
        best_model = model
        skf_regression_best={}
        
        for epoch in range(epoch_num):
            logger.info(f'\nEpoch [{epoch + 1}/{epoch_num}]')
            
            # Calculate prediction results and losses
            train_trues, train_scores, train_loss = cal_by_epoch(
                mode='train', model=model, loader=train_loader, criterion=criterion, optimizer=optimizer)
            
            val_trues, val_scores, val_loss = cal_by_epoch(
                mode='val', model=model, loader=val_loader, criterion=criterion)
            
            test_trues, test_scores, test_loss = cal_by_epoch(
                mode='test', model=model, loader=test_loader, criterion=criterion)

            # Calculate evaluation metrics
            if task_type == 'classification':
                train_metrics = cal_metrics(train_trues, train_scores, threshold)
                val_metrics = cal_metrics(val_trues, val_scores, threshold)
                test_metrics = cal_metrics(test_trues, test_scores, threshold)
                
                # 使用AUC作为评估指标
                val_metric = val_metrics[-2]  # AUC
                test_metric = test_metrics[-2]  # AUC
            else:  # regression
                train_metrics = cal_regression_metrics(train_trues, train_scores)
                val_metrics = cal_regression_metrics(val_trues, val_scores)
                test_metrics = cal_regression_metrics(test_trues, test_scores)
                
                # 使用R2作为评估指标
                val_metric = val_metrics['R2']
                test_metric = test_metrics['R2']

            # Print evaluation result
            if task_type == 'classification':
                logger.info(print_metrics('train', train_loss, train_metrics))
                logger.info(print_metrics('valid', val_loss, val_metrics))
            else:
                logger.info(print_regression_metrics('train', train_metrics))
                logger.info(print_regression_metrics('valid', val_metrics))
                # 添加更详细的回归任务日志
                logger.info(f"Validation R2: {val_metrics['R2']:.4f}, MSE: {val_metrics['MSE']:.4f}, MAE: {val_metrics['MAE']:.4f}")
                logger.info(f"Test R2: {test_metrics['R2']:.4f}, MSE: {test_metrics['MSE']:.4f}, MAE: {test_metrics['MAE']:.4f}")

            # Save the model by metric
            if val_metric > best_val_metric:
                count = 0
                best_model = model
                best_val_metric = val_metric
                best_test_metric = test_metric
                best_test_scores = test_scores
                logger.info(f"!!!Get better model with valid {task_type} metric:{val_metric:.6f}. ")
                if task_type == 'regression':
                    skf_regression_best=val_metrics
                    skf_regression_best["r"]=np.corrcoef(val_trues, val_scores)[0,1]
                    logger.info(f"Corresponding test R2: {test_metric:.6f}, r:{skf_regression_best['r']:.6f}")
            else:
                count += 1
                if count >= patience:
                    logger.info(f'Fold {i + 1} training done!!!\n')
                    break
        if task_type == 'regression':
            if "r" not in regression_best:
                regression_best=skf_regression_best
                final_best_model=best_model
            else:
                if skf_regression_best["r"]>regression_best["r"]:
                    regression_best=skf_regression_best
                    final_best_model=best_model
        if task_type == 'classification':
            if best_val_metric>max_auc:
                max_auc=best_val_metric
                final_best_model=best_model
        
        if best_val_metric <= 0.6 and task_type == 'classification':
            kfold = kfold - 1
            continue
            
        all_best_model.append(best_model)
        val_metrics_list.append(best_val_metric)
        test_metrics_list.append(best_test_metric)
        kfold_test_scores.append(best_test_scores)
    
    logger.info(f'Model training done!!!\n')
    for i, (val_metric, test_metric) in enumerate(zip(val_metrics_list, test_metrics_list)):
        logger.info(f'Fold {i + 1}: val metric:{val_metric:.6f}, test metric:{test_metric:.6f}.')
    
    # Average models' results
    final_test_scores = np.sum(np.array(kfold_test_scores), axis=0) / kfold
    
    if task_type == 'classification':
        # Cal the best threshold
        best_acc_threshold, best_acc = best_acc_thr(test_trues, final_test_scores)
        print(f'The best acc threshold is {best_acc_threshold:.2f} with the best acc({best_acc:.3f}).')
        logger.info(f'The best acc threshold is {best_acc_threshold:.2f} with the best acc({best_acc:.3f}).')
        
        # Select the best threshold by acc
        final_metrics = cal_metrics(test_trues, final_test_scores, best_acc_threshold)
        logger.info(print_metrics('Final test', test_loss, final_metrics))
    else:
        final_metrics = cal_regression_metrics(test_trues, final_test_scores)
        logger.info(print_regression_metrics('Final test', final_metrics))
        # 添加更详细的最终测试结果分析
        logger.info(f"Final Test Results Analysis:")
        logger.info(f"Mean of true values: {np.mean(test_trues):.4f}")
        logger.info(f"Mean of predicted values: {np.mean(final_test_scores):.4f}")
        logger.info(f"Std of true values: {np.std(test_trues):.4f}")
        logger.info(f"Std of predicted values: {np.std(final_test_scores):.4f}")
        logger.info(f"Correlation between true and predicted: {np.corrcoef(test_trues, final_test_scores)[0,1]:.4f}")
    
    if result_path != None:
        if task_type == 'classification':
            write_result(str(F_len), 'one-hot', args.__dict__, gene_name, final_metrics, result_path)
        else:
            write_regression_result(str(F_len), 'one-hot', args.__dict__, gene_name, final_metrics, result_path)
    # save_model_structures(model, gene_name, model_structures_path)
    if save_path != None:
        if task_type == 'classification':
            torch.save(final_best_model, save_path + "/" + gene_name + "_AUC:" + str(max_auc)[:6] + "_threshold:" + str(best_acc_threshold)[:6] + ".pkl")
            torch.save(final_best_model.state_dict(), save_path + "/" + gene_name + "_AUC:" + str(max_auc)[:6] + "_threshold:" + str(best_acc_threshold)[:6] + ".pth")
        else:
            torch.save(final_best_model, save_path + "/" + gene_name + "_R2:" + str(regression_best['R2'])[:6] + ".pkl")
            torch.save(final_best_model.state_dict(), save_path + "/" + gene_name + "_R2:" + str(regression_best['R2'])[:6] + ".pth")

def cal_by_epoch(mode, model, loader, criterion, optimizer=None):
    # Model on train mode
    model.train() if mode == 'train' else model.eval()
    all_trues, all_scores = [], []
    losses, sample_num = 0.0, 0
    
    for batch_idx, (X, features, FX, protein_features, protein_table_features, gnn_f, dnashape_f, inputs_f, attention_mask_f, y, fitness, raw_protein_seqs) in enumerate(loader):
        sample_num += len(y)

        # Create variables
        with torch.no_grad():
            X_var = torch.autograd.Variable(X.to(device).float())
            protein_features_var = torch.autograd.Variable(protein_features.to(device).float())
            protein_table_features_var = torch.autograd.Variable(protein_table_features.to(device).float())
            features_var = torch.autograd.Variable(features.to(device).float())
            FX_var = torch.autograd.Variable(FX.to(device).float())
            
            if model.task_type == 'classification':
                y_var = torch.autograd.Variable(y.to(device).float())
            else:  # regression
                y_var = torch.autograd.Variable(fitness.to(device).float())
            
            if args.us_DNASHAPE:
                dnashape_f = torch.tensor(dnashape_f).to(device)
                dnashape_f_var = torch.autograd.Variable(dnashape_f.to(device).float())
            else:
                dnashape_f_var = None
                
            if args.us_GNN:
                gnn_data = gnn_data.to(device)
            else:
                gnn_data = None
                
            if args.us_Pretraining:
                inputs_f_var = torch.autograd.Variable(inputs_f.to(device))
                attention_mask_f_var = torch.autograd.Variable(attention_mask_f.to(device))
            else:
                inputs_f_var = None
                attention_mask_f_var = None

        # compute output
        output = model(X_var, feature=features_var, fx=FX_var, gnn_data=gnn_data,
                      sf=dnashape_f_var, inputs_f=inputs_f_var,
                      attention_mask_f=attention_mask_f_var,
                      protein_features=protein_features_var,
                      protein_table_features=protein_table_features_var)
        
        # calculate loss
        loss = criterion(output.view(-1), y_var)
        
        # accumulate loss
        losses += loss.item()

        # compute gradient and do SGD step when training
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_trues.append(y_var.data.cpu().numpy())
        all_scores.append(output.data.cpu().numpy())
    
    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    
    # 确保数据是一维数组
    all_trues = np.array(all_trues).flatten()
    all_scores = np.array(all_scores).flatten()
    
    # 确保数据类型为float64
    all_trues = all_trues.astype(np.float64)
    all_scores = all_scores.astype(np.float64)
    
    if mode == 'val' and model.task_type == 'regression':
        # 添加可视化
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # 计算相关系数
        pearson_corr, pearson_p = stats.pearsonr(all_trues, all_scores)
        spearman_corr, spearman_p = stats.spearmanr(all_trues, all_scores)
        
        # 创建图形
        plt.figure(figsize=(10, 5))
        
        # 散点图
        plt.subplot(1, 2, 1)
        plt.scatter(all_trues, all_scores, alpha=0.5)
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Prediction\nPearson r={pearson_corr:.3f}, p={pearson_p:.3f}')
        
        # 添加回归线
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_trues, all_scores)
        x = np.linspace(min(all_trues), max(all_trues), 100)
        y = slope * x + intercept
        plt.plot(x, y, 'r--', label=f'y = {slope:.3f}x + {intercept:.3f}')
        plt.legend()
        
        # 残差图
        plt.subplot(1, 2, 2)
        residuals = all_scores - all_trues
        plt.scatter(all_trues, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Value')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot\nSpearman ρ={spearman_corr:.3f}, p={spearman_p:.3f}')
        
        plt.tight_layout()
        plt.savefig('prediction.png')
        plt.close()
        
        # 打印相关系数
        logger.info(f'Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.3f})')
        logger.info(f'Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3f})')
    
    # Calculate average loss
    avg_loss = losses / sample_num
    
    return all_trues, all_scores, avg_loss

def get_result_gene(file_path):
    data=list(pd.read_csv(file_path)["Model_structure"])
    data=[x.split(":")[-1] for x in data]
    return data

if __name__ == '__main__':
    # 对S1数据集进行训练
    label='label'
    train_cv(gene_name=label,fitness=None,DSEG_label=None,result_path=args.result_path,model_structures_path="result/Model_structures.txt",save_path=args.save_path)