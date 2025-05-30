import argparse
import json
import sys, os
import torch

class ATFNetConfigs:
    def __init__(self, max_len, fea_size,ATFNet_output,d_model,factor,n_heads,e_layers,d_ff,atf_drop,fnet_d_ff,fnet_d_model,complex_dropout,fnet_layers,is_emb):
        self.seq_len = max_len
        self.pred_len = ATFNet_output  # 96 192 336 720
        self.d_model = d_model  # 模型的维度
        self.factor = factor  # 用于缩放注意力机制的因子
        self.n_heads = n_heads  # 注意力头的数量 
        self.e_layers = e_layers  # 编码器的层数
        self.d_ff = d_ff  # 前馈神经网络的维度
        self.dropout = atf_drop  # dropout的概率
        self.activation = 'gelu'  # 激活函数
        self.enc_in = fea_size  # 编码器输入的特征数量
        self.dec_in = fea_size  # 解码器输入的特征数量
        self.c_out = 1  # 输出的特征数量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU还是CPU
        self.fnet_d_ff = fnet_d_ff  # 频域前馈神经网络的维度 
        self.fnet_d_model = fnet_d_model  # 频域模型的维度
        self.complex_dropout = complex_dropout  # 复数dropout的概率
        self.fnet_layers = fnet_layers  # 频域网络的层数
        self.is_emb = is_emb  # 是否使用嵌入层
        self.output_attention = True # 是否输出注意力

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def nucleotide_params(parser):
    # Note
    parser.add_argument('--note', type=str, default="none",help="Record training remarks")
    # Train setting
    parser.add_argument('--is_balance_data', type=str2bool, nargs='?', default=False,help="Set to True to balance the dataset, False otherwise")
    parser.add_argument('--is_ATFnet', type=str2bool, nargs='?', default=True,help="Set to True to use ATFnet, False otherwise")
    parser.add_argument('--is_motif_extraction', type=str2bool, nargs='?', default=False,help="Set to True to extract motif, False otherwise")
    parser.add_argument('--task_type', type=str, default="classification",help="classification or regression")
    parser.add_argument('--is_esm', type=str2bool, nargs='?', default=False,help="Set to True to use ESM, False otherwise")
    parser.add_argument('--us_protein_table', type=str2bool, nargs='?', default=False,help="Set to True to use protein table, False otherwise")

    # File Path Setting
    parser.add_argument('--dataset_path', type=str, default="data/S4.csv",help="Specify the path to the dataset (default: data/S1.csv)")
    parser.add_argument('--fitness_path', type=str, default="data/S4_fitness.csv",help="Specify the path to the fitness")
    parser.add_argument('--dnashape_path', type=str, default="data/DNASHAPE/S4shapefeature/",help="Specify the path to the dataset")
    parser.add_argument('--protein_feature_path', type=str, default="data/S4_feature/protein/protein_feature.csv",help="Specify the path to the dataset")
    parser.add_argument('--feature_path', type=str, default="data/S4_feature/final/feature5000.csv",help="Specify the path to the dataset")
    parser.add_argument('--pretraining_path', type=str, default="GROVER_PreTrain",help="Pre-training model paths")
    parser.add_argument('--pretraining_name', type=str, default="GROVER_PreTrain",help="Pre-training model name")
    parser.add_argument('--result_path', type=str, default="result/grids.csv",help="result paths")
    parser.add_argument('--save_path', type=str, default="saved_model/curmodel",help="model save paths")


    # model architecture
    parser.add_argument('--is_double_stranded', type=str2bool, nargs='?',default=False,help="Set to True for double-stranded DNA, False for single-stranded DNA")
    parser.add_argument('--is_only_feature',type=str2bool, nargs='?',default=False,help="Set to True for only using the feature, False for using the feature and sequence")
    parser.add_argument('--us_GNN', type=str2bool, nargs='?',default=False,help="Set to True for using GNN, False for not using GNN")
    parser.add_argument('--us_DNASHAPE',type=str2bool,nargs='?',default=False,help="Set to True for using Dnashape,False for not using Dnashape")
    parser.add_argument('--us_Pretraining',type=str2bool,nargs='?',default=False,help="Whether to use pre-trained models")
    parser.add_argument('--us_protein',type=str2bool,nargs='?',default=True,help="Whether to use protein")

    parser.add_argument('--seq_type', type=str, default="nucleotide")
    parser.add_argument('--max_len', type=int, default=1800)
    parser.add_argument('--protein_max_len', type=int, default=600)
    parser.add_argument('--emb_type', type=str, default="onehot")
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=114514)

    # model-specific parameters
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--head_num', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--lstm_drop', type=float, default=0.0)
    parser.add_argument('--linear_drop', type=float, default=0.2)

    # ATFNet configs
    parser.add_argument('--d_model', type=int, default=1) # 模型维度
    parser.add_argument('--factor', type=int, default=2) # 用于缩放注意力机制的因子

    parser.add_argument('--n_heads', type=int, default=1) # 注意力头的数量 Max 2
    parser.add_argument('--e_layers', type=int, default=1) # 编码器的层数 Max 1
    parser.add_argument('--d_ff', type=int, default=4) # 前馈神经网络的维度 Max 128

    parser.add_argument('--atf_drop', type=float, default=0.1) # dropout 的概率
    parser.add_argument('--fnet_d_ff', type=int, default=8) # 频域前馈神经网络的维度 Max 128
    parser.add_argument('--fnet_d_model', type=int, default=8) # 频域模型的维度 Max 128
    parser.add_argument('--complex_dropout', type=float, default=0.0) # 复数dropout 的概率
    parser.add_argument('--fnet_layers', type=int, default=1) # 频域模型的层数
    parser.add_argument('--is_emb',type=str2bool,nargs='?',default=True,help="Whether to use embedded layers")


    # learning process parameters
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32) # 128
    parser.add_argument('--patience', type=int, default=10) # 等待多少轮次
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0.0001) # 正则化参数
    parser.add_argument('--threshold', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    # args.save_path = os.path.join(os.path.curdir, 'saved_model', args.seq_type)
    args.log_path=os.path.join(os.path.curdir, 'log', args.seq_type)
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)
    return args

def set_params():
    argv = sys.argv
    parser = argparse.ArgumentParser()
    args = nucleotide_params(parser)
    with open(os.path.join(args.save_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args