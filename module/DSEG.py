from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
import math
from torch.nn.functional import relu
from math import sqrt
from module import ATFNet
from transformers import AutoModel
from utils.load_bigmodel import BigPreTrainModel


model_output_len={
    'hyenadna-tiny-16k-seqlen-d128-hf': 512,
    'nucleotide-transformer-500m-human-ref': 512,
    'DNA_bert_3': 512,
    'splicebert.510nt': 510,
    'nucleotide-transformer-v2-50m-multi-species': 512,
    'nucleotide-transformer-2.5b-multi-species': 512,
    'nucleotide-transformer-v2-250m-multi-species': 512,
    'DNA_bert_4': 512,
    'nucleotide-transformer-2.5b-1000g': 512,
    'hyenadna-tiny-1k-seqlen-d256-hf': 512,
    'nucleotide-transformer-500m-1000g': 512,
    'nucleotide-transformer-v2-500m-multi-species': 512,
    'splicebert-human.510nt': 510,
    'hyenadna-tiny-1k-seqlen-hf': 512,
    'hyenadna-medium-160k-seqlen-hf': 512,
    'agro-nucleotide-transformer-1b': 512,
    'splicebert': 510,
    'GROVER': 512,
    'hyenadna-medium-450k-seqlen-hf': 512,
    'DNA_bert_6': 512,
    'DNA_bert_5': 512,
    'nucleotide-transformer-v2-100m-multi-species': 512,
    'gena-lm-bert-base-t2t': 512,
    'caduceus-ph_seqlen-131k_d_model-256_n_layer-16': 512,
    'hyenadna-small-32k-seqlen-hf': 512,
    'caduceus-ps_seqlen-131k_d_model-256_n_layer-16': 512,
    'DNABERT-2-117M':512,
    'DNABERT-2-117M-3256':512,
    "DNABERT2-ME":512
 }

class SequenceProcessor(nn.Module):
    def __init__(self, input_dim, kernel_size, num_head, hidden_size, num_layers, attn_drop, lstm_drop, is_ATFnet, config=None,type="DNA"):
        super(SequenceProcessor, self).__init__()
        self.is_ATFnet = is_ATFnet
        self.type = type
        
        # CNN层
        self.cnn = nn.Conv1d(in_channels=input_dim,
                           out_channels=input_dim,
                           kernel_size=kernel_size,
                           padding='same')
        
        # 注意力机制
        if is_ATFnet and type=="DNA":
            self.attention = ATFNet.Model(config).to(config.device)
        else:
            self.attention = nn.MultiheadAttention(embed_dim=input_dim,
                                                 num_heads=num_head,
                                                 dropout=attn_drop,
                                                 batch_first=True)
            self.layer_norm = nn.LayerNorm(input_dim)
        
        # BiLSTM层
        self.bilstm = nn.LSTM(input_dim,
                            hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=lstm_drop)
        
        # 池化层
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        # 残差连接和CNN处理
        residual = x
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        x = residual + x.permute(0, 2, 1)
        # print(self.type,self.is_ATFnet)
        # 注意力机制
        if self.is_ATFnet and self.type=="DNA":
            x, attention = self.attention(x)
        else:
            attn_output, _ = self.attention(x, x, x)
            x = x + self.layer_norm(attn_output)
            attention = None
        
        # BiLSTM处理
        x, _ = self.bilstm(x)
        x = self.pool(x).squeeze(-1)
        
        return x, attention

class DSEG(nn.Module):
    def __init__(self, max_len, fea_size, protein_max_len, protein_fea_size,
                 kernel_size, num_head, hidden_size, num_layers, attn_drop, lstm_drop,
                 linear_drop, atfconfigs, shape_atf_config,
                 F_len, is_double_stranded, is_only_feature, us_gnn=False, us_DNASHAPE=False, 
                 us_Pretraining=False, pretraining_path=None, is_ATFnet=True, is_motif_extraction=False,
                 us_protein=False, task_type='classification',  # 新增参数：task_type可以是'classification'或'regression'
                 structure='TextCNN+MultiheadAttn+BiLSTM+Maxpool+MLP', name='DSEG',
                 big_model:Optional[BigPreTrainModel]=None,is_esm=False,us_protein_table=True):
        super(DSEG, self).__init__()

        self.protein_max_len = protein_max_len
        self.ATFNet_output = max_len
        self.config = atfconfigs
        self.task_type = task_type  # 保存任务类型
        

        self.structure = structure
        self.name = name
        self.is_double_stranded = is_double_stranded
        self.is_motif_extraction = is_motif_extraction
        self.is_only_feature = is_only_feature
        self.us_gnn = us_gnn
        self.us_DNASHAPE = us_DNASHAPE
        self.us_Pretraining = us_Pretraining
        self.us_protein = us_protein
        self.us_protein_table = us_protein_table
        self.hidden_channels = 64
        self.max_len = max_len
        self.is_ATFnet = is_ATFnet
        self.big_model = big_model
        self.is_esm = is_esm
        # 保存中间输出
        self.intermediate_outputs = {}

        # 序列处理模块
        self.dna_processor = SequenceProcessor(
            input_dim=fea_size,
            kernel_size=kernel_size,
            num_head=num_head,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_drop=attn_drop,
            lstm_drop=lstm_drop,
            is_ATFnet=is_ATFnet,
            config=atfconfigs,
            type="DNA"
        )
        
        if us_protein:
            atfconfigs.d_model = protein_fea_size
            atfconfigs.seq_len = protein_max_len
            self.protein_processor = SequenceProcessor(
                input_dim=protein_fea_size,
                kernel_size=kernel_size,
                num_head=num_head,
                hidden_size=hidden_size,
                num_layers=num_layers,
                attn_drop=attn_drop,
                lstm_drop=lstm_drop,
                is_ATFnet=is_ATFnet,
                config=atfconfigs,
                type="protein"
            )
        
        if self.is_double_stranded:
            self.reverse_processor = SequenceProcessor(
                input_dim=fea_size,
                kernel_size=kernel_size,
                num_head=num_head,
                hidden_size=hidden_size,
                num_layers=num_layers,
                attn_drop=attn_drop,
                lstm_drop=lstm_drop,
                is_ATFnet=is_ATFnet,
                config=atfconfigs
            )
        
        if self.us_DNASHAPE:
            self.shape_processor = SequenceProcessor(
                input_dim=14,
                kernel_size=kernel_size,
                num_head=num_head,
                hidden_size=hidden_size,
                num_layers=num_layers,
                attn_drop=attn_drop,
                lstm_drop=lstm_drop,
                is_ATFnet=is_ATFnet,
                config=shape_atf_config
            )
        
        if self.us_Pretraining:
            self.pretrain_pool = nn.AdaptiveAvgPool1d(1)
            self.pretrain_pool2 = nn.AdaptiveAvgPool2d(1)
            self.pretrain_linear = nn.Linear(model_output_len[self.big_model.name], 256)
        
        if self.us_gnn:
            self.gat1 = GATv2Conv(4, self.hidden_channels)
            self.gat2 = GATv2Conv(self.hidden_channels, self.hidden_channels)
            self.gat3 = GATv2Conv(self.hidden_channels, self.hidden_channels)
            self.gcn1 = GCNConv(self.hidden_channels, self.hidden_channels)
            self.gcn2 = GCNConv(self.hidden_channels, self.hidden_channels)
            self.highway = nn.Linear(self.hidden_channels, self.hidden_channels)
        
        if self.is_esm:
            self.esm_model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
            self.batch_converter = self.alphabet.get_batch_converter()
        
        final_input_len = self._calculate_final_input_len(F_len)
        
        self.shared_layers = nn.Sequential(
            nn.Linear(final_input_len, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(linear_drop)
        )
        self.protein_shared_layers = nn.Sequential(
            nn.Linear(protein_max_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(linear_drop)
        )
        
        # 根据任务类型选择输出层
        if task_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )
        else:  # regression
            self.output_head = nn.Sequential(
                nn.Linear(1024, 1)
            )

    def build_adj_matrix(self, edge_index, num_nodes):
        """从edge_index构建邻接矩阵"""
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        row, col = edge_index
        adj_matrix[row, col] = 1
        return adj_matrix
    
    def adj_to_edge_index(self, adj_matrix):
        """将邻接矩阵转换回edge_index形式"""
        row, col = adj_matrix.nonzero(as_tuple=True)
        return torch.stack([row, col], dim=0)
    # x:primary sequence
    # feature:features
    # fx:inverse complementary sequence
    # gnn_data:graph neural network
    # sf:shape feature
    def forward(self, x,feature=False, fx=None, gnn_data=None, sf=None, get_attn=False, 
                inputs_f=None, attention_mask_f=None, seq=None, protein_features=None, raw_protein_seqs=None,protein_table_features=None):
        if self.is_only_feature:
            shared_features = self.shared_layers(feature)
            output = self.output_head(shared_features)
            return output
        # 处理DNA序列
        x, dna_attention = self.dna_processor(x)
        
        if self.us_protein:
            protein_features, protein_attention = self.protein_processor(protein_features)
            
            # 如果启用了ESM模型，处理原始蛋白质序列
            if self.is_esm:
                esm_features = self._process_esm(raw_protein_seqs)
                if esm_features is not None:
                    # 将ESM特征与其他蛋白质特征相加而不是拼接
                    protein_features = protein_features + esm_features
        
        if self.is_double_stranded:
            fx, fx_attention = self.reverse_processor(fx)
        
        if self.us_DNASHAPE:
            sf, sf_attention = self.shape_processor(sf)
        
        if self.us_gnn:
            gnn_x = self._process_gnn(gnn_data)
        
        if self.us_Pretraining:
            pre_output = self._process_pretraining(inputs_f, attention_mask_f)
        
        # 特征拼接
        features = [x]
        self.intermediate_outputs['dna_features'] = x
        if self.us_protein:
            features.append(protein_features)
            self.intermediate_outputs['protein_features'] = protein_features
        if not self.is_motif_extraction:
            features.append(feature)
            self.intermediate_outputs['feature'] = feature
        if self.us_protein_table:
            features.append(protein_table_features)
            self.intermediate_outputs['protein_table_features'] = protein_table_features
        if self.us_gnn:
            features.append(gnn_x)
            self.intermediate_outputs['gnn_x'] = gnn_x
        if self.is_double_stranded:
            features.append(fx)
            self.intermediate_outputs['fx'] = fx
        if self.us_DNASHAPE:
            features.append(sf)
            self.intermediate_outputs['sf'] = sf
        if self.us_Pretraining and not self.is_motif_extraction:
            features.append(pre_output)
            self.intermediate_outputs['pre_output'] = pre_output
        # print("features",len(features))
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)
        x = torch.cat(features, dim=1)
        self.intermediate_outputs['x'] = x
        # 通过共享层
        shared_features = self.shared_layers(x)
        self.intermediate_outputs['shared_features'] = shared_features
        # 输出层
        output = self.output_head(shared_features)
        
        if get_attn:
            return output, dna_attention
        return output
    
    def _process_gnn(self, gnn_data):
        node_features, edge_index, batch = gnn_data.x, gnn_data.edge_index, gnn_data.batch
        adj_matrix = self.build_adj_matrix(edge_index, node_features.size(0))
        
        # 第一层GAT + GCN
        x1 = F.relu(self.gat1(node_features, edge_index))
        x1 = F.relu(self.gcn1(x1, edge_index))
        
        # 第二层GAT + GCN
        adj_matrix_squared = torch.matmul(adj_matrix, adj_matrix)
        edge_index_squared = self.adj_to_edge_index(adj_matrix_squared)
        x2 = F.relu(self.gat2(node_features, edge_index_squared))
        x2 = F.relu(self.gcn2(x2, edge_index_squared))
        
        # 第三层GAT
        adj_matrix_cubed = torch.matmul(adj_matrix_squared, adj_matrix)
        edge_index_cubed = self.adj_to_edge_index(adj_matrix_cubed)
        x3 = F.relu(self.gat3(node_features, edge_index_cubed))
        
        # Highway连接
        x_highway = self.highway(x1 + x2 + x3)
        
        # 全局平均池化
        return global_mean_pool(x_highway, batch)
    
    def _process_pretraining(self, inputs_f, attention_mask_f):
        if "hyenadna" in self.big_model.name:
            pre_output = self.big_model.get_model_output(inputs_f, attention_mask=attention_mask_f)
        else:
            pre_output = self.big_model.get_model_output(inputs_f, attention_mask_f)
            
        if pre_output.ndimension() == 3:
            pre_output = self.pretrain_pool(pre_output).squeeze(-1)
        elif pre_output.ndimension() == 4:
            pre_output = self.pretrain_pool2(pre_output).squeeze(-1)
            pre_output = self.pretrain_pool(pre_output).squeeze(-1)
            
        return self.pretrain_linear(pre_output)

    def _process_esm(self, raw_protein_seqs):
        """处理蛋白质序列通过ESM模型"""
        if not self.is_esm or raw_protein_seqs is None:
            return None
            
        # 准备批处理数据
        batch_labels, batch_strs, batch_tokens = [], [], []
        for i, seq in enumerate(raw_protein_seqs):
            batch_labels.append(str(i))
            batch_strs.append(seq)
        
        # 使用batch_converter处理序列
        batch_labels, batch_strs, batch_tokens = self.batch_converter(list(zip(batch_labels, batch_strs)))
        
        # 将数据移动到正确的设备
        batch_tokens = batch_tokens.to(next(self.esm_model.parameters()).device)
        
        # 使用no_grad来提高推理效率
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33])
            representations = results["representations"][33]
        
        # 使用平均池化获取序列表示
        # 去掉特殊token (CLS, EOS等)的表示
        protein_representations = representations[:, 1:representations.shape[1]-1].mean(dim=1)
        
        # 添加一个投影层来调整维度
        if not hasattr(self, 'esm_projection'):
            self.esm_projection = nn.Linear(1280, self.protein_max_len).to(batch_tokens.device)
        
        protein_representations = self.esm_projection(protein_representations)
        return protein_representations

    def _calculate_final_input_len(self, F_len):
        final_input_len = 0
        if self.is_only_feature:
            final_input_len = F_len
        else:
            if self.is_double_stranded:
                if self.us_protein:
                    # ESM特征现在与protein_features相加而不是拼接
                    final_input_len = self.ATFNet_output * 2 + F_len + self.protein_max_len
                else:
                    final_input_len = self.ATFNet_output * 2 + F_len
            else:
                if self.us_protein:
                    # ESM特征现在与protein_features相加而不是拼接
                    final_input_len = self.ATFNet_output + F_len + self.protein_max_len
                else:
                    final_input_len = self.ATFNet_output + F_len
        
        if self.us_gnn:
            final_input_len += self.hidden_channels
        if self.us_DNASHAPE:
            final_input_len += self.ATFNet_output
        if self.us_Pretraining:
            final_input_len += 256
        if self.is_motif_extraction:
            final_input_len -= F_len
        # if self.is_ATFnet:
            # final_input_len += 1200 # 补上之前缺失的蛋白质 ATFNet 的输出
        if self.us_protein_table:
            final_input_len += 1280
        return final_input_len

    def get_task_weights(self):
        """获取归一化的任务权重"""
        weights = F.softmax(self.task_weights, dim=0)
        return weights
