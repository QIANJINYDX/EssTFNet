import os
import pickle, torch
from typing import Optional
import random

import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.load_bigmodel import BigPreTrainModel

def readspots(path):
    with open(path, "r") as f:
        text = f.readlines()
    ans = []
    current = []
    spot = ""
    for txt in text:
        if ">" in txt:
            if len(current) != 0:
                current.append(spot)
                ans.append(current)
            current = []
            spot = ""
            # add name
            current.append(txt.replace("\n", ''))
        else:
            spot = spot + txt.replace("\n", '')
    current.append(spot)
    ans.append(current)
    return ans


def seq_process(seq, max_len):
    """
    :param seq: 序列
    :param max_len:序列填充或截断长度
    :return: 处理之后的序列
    """
    if len(seq) < max_len:
        return seq + '*' * (max_len - len(seq))
    else:
        return seq[:max_len]


def seqItem2id(item, seq_type):
    """ 将序列转换为token
    :param item:
    :param seq_type: 蛋白质 or DNA
    :return:
    """
    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'protein' else 'ATCG'
    seqItem2id = {}
    seqItem2id.update(dict(zip(items, range(1, len(items) + 1))))
    seqItem2id.update({"*": 0})
    return seqItem2id[item]


def id2seqItem(i, seq_type):
    """ 将token转换为序列
    :param i:
    :param seq_type:
    :return:
    """
    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'protein' else 'ATCG'
    id2seqItem = ["*"] + list(items)
    return id2seqItem[i]


def vectorize(emb_type, seq_type, window=13, sg=1, workers=8):
    """ Get embedding of 'onehot' or 'word2vec-[dim] """
    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'protein' else 'ATCG'
    emb_path = os.path.join(r'../embeds/', seq_type)
    emb_file = os.path.join(emb_path, emb_type + '.pkl')

    # if os.path.exists(emb_file):
    #     with open(emb_file, 'rb') as f:
    #         embedding = pickle.load(f)
    #     # print(f'Loaded cache from {emb_file}.')
    #     return embedding

    if emb_type == 'onehot':
        embedding = np.concatenate(([np.zeros(len(items))], np.eye(len(items)))).astype('float32')
        # embedding=np.array([[0,0,0],[-0.5,0.5,0.5],[0.5,-0.5,0.5],[-0.5,-0.5,-0.5],[0.5,0.5,-0.5]]).astype(np.float32)
    elif emb_type[:8] == 'word2vec':
        _, emb_dim = emb_type.split('-')[0], int(emb_type.split('-')[1])
        seq_data, _, _ = load_data("data/S1.fasta")
        # seq_data = pickle.load(open(r'../data/seq_data.pkl', 'rb'))[seq_type]
        doc = [list(i) for i in list(seq_data)]
        model = Word2Vec(doc, min_count=1, window=window, vector_size=emb_dim, workers=workers, sg=sg, epochs=10)
        char2vec = np.zeros((len(items) + 1, emb_dim))
        for i in range(len(items) + 1):
            if id2seqItem(i, seq_type) in model.wv:
                char2vec[i] = model.wv[id2seqItem(i, seq_type)]
        embedding = char2vec

    if os.path.exists(emb_path) == False:
        os.makedirs(emb_path)
    with open(emb_file, 'wb') as f:
        pickle.dump(embedding, f, protocol=4)
    # print(f'Loaded cache from {emb_file}.')
    return embedding


class CelllineDataset(Dataset):
    def __init__(self, indexes, seqs,protein_seqs,labels,fitness, table_features,protein_table_features,fseqs,emb_type, seq_type, max_len,protein_max_len,gnn_features=None,dnashape=None,tokens_ids=None,attention_mask=None):
        self.indexes = indexes
        self.labels = labels
        self.fitness = fitness
        self.num_ess = np.sum(self.labels == 1)
        self.num_non = np.sum(self.labels == 0)
        self.raw_seqs = seqs
        self.raw_protein_seqs = protein_seqs
        self.processed_seqs = [seq_process(seq, max_len) for seq in self.raw_seqs]
        self.processed_protein_seqs = [seq_process(seq, protein_max_len) for seq in self.raw_protein_seqs]
        self.tokenized_seqs = [[seqItem2id(i, seq_type) for i in seq] for seq in self.processed_seqs]
        self.tokenized_protein_seqs = [[seqItem2id(i, 'protein') for i in seq] for seq in self.processed_protein_seqs]

        embedding = nn.Embedding.from_pretrained(torch.tensor(vectorize(emb_type, seq_type)))
        protein_embedding = nn.Embedding.from_pretrained(torch.tensor(vectorize('onehot', 'protein')))
        self.protein_emb_dim = protein_embedding.embedding_dim
        self.emb_dim = embedding.embedding_dim
        if table_features!=None:
            self.table_features = torch.FloatTensor(table_features)
        if protein_table_features!=None:
            self.protein_table_features = torch.FloatTensor(protein_table_features)
        self.features = embedding(torch.LongTensor(self.tokenized_seqs))
        self.protein_features = protein_embedding(torch.LongTensor(self.tokenized_protein_seqs))
        self.fprocessed_seqs = [seq_process(seq, max_len) for seq in fseqs]
        self.ftokenized_seqs = [[seqItem2id(i, seq_type) for i in seq] for seq in self.fprocessed_seqs]
        self.ffeatures = embedding(torch.LongTensor(self.ftokenized_seqs))

        if gnn_features!=None:
            self.gnn_features=gnn_features
        else:
            self.gnn_features=list(range(len(labels)))
        if dnashape!=None:
            self.dnashape=dnashape
        else:
            self.dnashape=list(range(len(labels)))
        if tokens_ids!=None:
            self.input_ids=tokens_ids
        else:
            self.input_ids=list(range(len(labels)))
            
        if attention_mask!=None:
            self.attention_mask=attention_mask
        else:
            self.attention_mask=list(range(len(labels)))
        # print(type(self.features),type(self.table_features),type(self.ffeatures),type(self.protein_features),type(self.protein_table_features),type(self.gnn_features),type(self.dnashape),type(self.input_ids),type(self.attention_mask),type(self.labels),type(self.fitness),type(self.raw_protein_seqs))
        # print("features",self.features.shape,"table_features",self.table_features.shape,"ffeatures",self.ffeatures.shape,"protein_features",self.protein_features.shape,"protein_table_features",self.protein_table_features.shape,"gnn_features",len(self.gnn_features),"dnashape",len(self.dnashape),"input_ids",len(self.input_ids),"attention_mask",len(self.attention_mask),"labels",self.labels.shape,"fitness",self.fitness.shape,"raw_protein_seqs",len(self.raw_protein_seqs))



    def __getitem__(self, item):
        # if self.gnn_features and self.dnashape:
        #     return self.features[item], self.table_features[item],self.ffeatures[item],self.labels[item],self.dnashape[item]
        # if self.gnn_features:
        #     return self.features[item], self.table_features[item],self.ffeatures[item],self.labels[item],self.features[item]
        # if self.dnashape:
        #     return self.features[item], self.table_features[item],self.ffeatures[item],self.labels[item],self.dnashape[item]
        # else:
        #     return self.features[item], self.table_features[item],self.ffeatures[item],self.labels[item],self.features[item]
            
        return self.features[item], self.table_features[item],self.ffeatures[item],self.protein_features[item],self.protein_table_features[item],self.gnn_features[item],self.dnashape[item],self.input_ids[item],self.attention_mask[item],self.labels[item],self.fitness[item],self.raw_protein_seqs[item]

    def __len__(self):
        return len(self.indexes)


def readspots(path):
    with open(path, "r") as f:
        text = f.readlines()
    ans = []
    current = []
    spot = ""
    for txt in text:
        if ">" in txt:
            if len(current) != 0:
                current.append(spot)
                ans.append(current)
            current = []
            spot = ""
            # add name
            current.append(txt.replace("\n", ''))
        else:
            spot = spot + txt.replace("\n", '')
    current.append(spot)
    ans.append(current)
    return ans


def load_data(path):
    """

    :param path: 序列路径
    """
    fasta = readspots(path)
    seq_data = []
    label_data = []
    seq_id = []
    for i in range(len(fasta)):
        seq_data.append(fasta[i][1])
        seq_id.append(str(fasta[i][0]).split("|")[1])
        if "non" in fasta[i][0]:
            label_data.append(0)
        else:
            label_data.append(1)
    # print(seq_data,label_data,seq_id)
    return seq_data, label_data, seq_id


def Reverse(seq):
    res = ""
    for i in seq[::-1]:
        if i == "A":
            res += "T"
        elif i == "G":
            res += "C"
        elif i == "C":
            res += "G"
        elif i == "T":
            res += "A"
    return res

def load_data_csv(csv_path):
    data=pd.read_csv(csv_path)
    if "label" in list(data.columns):
        return list(data['protein']),list(data['nucleotide']),list(data['label'])
    else:
        return list(data['protein']),list(data['nucleotide']),None
def build_graph(positions, nucleotides, pairings):
    num_nodes = len(positions)
    x = torch.zeros((num_nodes, 4))
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    for i, nuc in enumerate(nucleotides):
        idx = nucleotide_to_index.get(nuc, -1)
        if idx >= 0:
            x[i, idx] = 1
        else:
            pass
    edge_index = []
    for i in range(1, num_nodes):
        edge_index.append([i - 1, i])
        edge_index.append([i, i - 1]) 
    for i, pairing in enumerate(pairings):
        if pairing > 0 and pairing - 1 != i:
            edge_index.append([i, pairing - 1])
            edge_index.append([pairing - 1, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data
def read_bpseq_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    positions = []
    nucleotides = []
    pairings = []

    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        pos = int(parts[0])
        nuc = parts[1]
        pairing = int(parts[2])

        positions.append(pos)
        nucleotides.append(nuc)
        pairings.append(pairing)

    return positions, nucleotides, pairings
def build_dataset(bpseq_folder, labels):
    dataset = []
    file_paths=os.listdir(bpseq_folder)
    file_paths=[x for x in file_paths if x.endswith('.bpseq')]
    file_paths.sort(key=lambda x:int(x[:-6]))
    for i in tqdm(range(len(file_paths))):
        filename = file_paths[i]
        if filename.endswith('.bpseq'):
            filepath = os.path.join(bpseq_folder, filename)
            positions, nucleotides, pairings = read_bpseq_file(filepath)
            data = build_graph(positions, nucleotides, pairings)
            if labels is not None:
                data.y = torch.tensor([labels[i]], dtype=torch.long)
                dataset.append(data)
    return dataset
def read_dnashape_feature(file_path,seq=",",max_len=500):
    feature=[]
    featurename=file_path.split('_')[1].split('.txt')[0]
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines=[line.strip() for line in lines]
    for i in range(len(lines)):
        if not lines[i].startswith('>'):
            cur=[float(x) for x in lines[i].split(seq)]
            if featurename in ["Shift",'Roll','Slide','Rise','Tilt','HelT']:
                cur.append(cur[len(cur)-1])
            feature.append(cur[:max_len])
    return feature

# 获取单个输入
def get_single_input(seq,protein_seq,table_features,emb_type,seq_type,max_len,protein_max_len,big_model:Optional[BigPreTrainModel]=None):
    test_indexes=[0 for _ in range(len(seq))]
    test_seqs=seq
    test_protein_seqs=protein_seq
    test_labels=[0 for _ in range(len(seq))]
    test_fitness=[0 for _ in range(len(seq))]
    table_features = table_features.values.tolist()
    ftest_seqs=[Reverse(i) for i in test_seqs]
    test_dataset = CelllineDataset(test_indexes, test_seqs, test_protein_seqs, test_labels, test_fitness, table_features,ftest_seqs,emb_type,seq_type, max_len,protein_max_len)

    return test_dataset


def load_dataset(seq_type, emb_type, max_len,protein_max_len,feature_path, seed,dataset_path,dnashape_path,pretraining_path,DSEG_label=None,us_GNN=False,us_DNASHAPE=False,us_Pretraining=False,is_balance_data=False,big_model:Optional[BigPreTrainModel]=None,is_motif_extraction=False,fitness=None,protein_feature_path=None):
    """ Load train & test dataset """
    # load dataset
    protein_seq_data, seq_data, label_data = load_data_csv(dataset_path)
    # print(protein_seq_data[0])
    # 自定义Label
    if DSEG_label!=None:
        label_data=DSEG_label
    if fitness!=None:
        fitness=fitness
    else:
        fitness=[0 for _ in range(len(label_data))]
    if is_motif_extraction==True:
        table_features = pd.DataFrame(index=range(len(label_data)))
    else:
        table_features = pd.read_csv(feature_path, header=0)
    if protein_feature_path!=None and os.path.exists(protein_feature_path):
        protein_table_features = pd.read_csv(protein_feature_path, header=0)
    else:
        protein_table_features = pd.DataFrame(index=range(len(label_data)))
    if 'label' in list(table_features.columns):
        table_features = table_features.drop(columns=["label"])
    col_len = len(table_features.columns)
    table_features = table_features.values.tolist()
    protein_table_features = protein_table_features.values.tolist()
    if is_balance_data:
        ess_indexes = [i for i, e in enumerate(label_data) if int(e) == 1]
        num_ess=len(ess_indexes)
        new_protein_seq_data,new_seq_data,new_label_data,new_table_features,new_protein_table_features,new_fitness_data=[],[],[],[],[],[]
        for i in range(len(label_data)):
            if label_data[i]==0 and num_ess>0:
                new_protein_seq_data.append(protein_seq_data[i])
                new_seq_data.append(seq_data[i])
                new_label_data.append(label_data[i])
                new_table_features.append(table_features[i])
                new_protein_table_features.append(protein_table_features[i])
                new_fitness_data.append(fitness[i])
                num_ess=num_ess-1
            elif label_data[i]==1:
                new_protein_seq_data.append(protein_seq_data[i])
                new_seq_data.append(seq_data[i])
                new_label_data.append(label_data[i])
                new_table_features.append(table_features[i])
                new_protein_table_features.append(protein_table_features[i])
                new_fitness_data.append(fitness[i])
        label_data=new_label_data
        fitness=new_fitness_data
        seq_data=new_seq_data
        table_features=new_table_features
        protein_seq_data=new_protein_seq_data
        protein_table_features=new_protein_table_features
    ess_indexes = [i for i, e in enumerate(label_data) if int(e) == 1]
    non_indexes = [i for i, e in enumerate(label_data) if int(e) == 0]
    num_ess = len(ess_indexes)
    num_non = len(non_indexes)
    # split data with balanced test set
    r = 0.2
    random.seed(seed)
    random.shuffle(ess_indexes)
    random.shuffle(non_indexes)
    test_indexes = ess_indexes[:int(num_ess * r)] + non_indexes[:int(num_ess * r)]
    train_indexes = list(set(ess_indexes + non_indexes) - set(test_indexes))

    # print(len(train_indexes),len(test_indexes),len(seq_data))

    train_seqs = [seq_data[i] for i in train_indexes]
    ftrain_seqs = [Reverse(i) for i in train_seqs]
    train_protein_seqs = [protein_seq_data[i] for i in train_indexes]

    train_labels = np.array([label_data[i] for i in train_indexes])
    train_fitness = np.array([fitness[i] for i in train_indexes])
    train_table_features = [table_features[i] for i in train_indexes]
    train_protein_table_features = [protein_table_features[i] for i in train_indexes]
    test_seqs = [seq_data[i] for i in test_indexes]
    ftest_seqs=[Reverse(i) for i in test_seqs]

    test_protein_seqs = [protein_seq_data[i] for i in test_indexes]
    test_labels = np.array([label_data[i] for i in test_indexes])
    test_fitness = np.array([fitness[i] for i in test_indexes])
    test_table_features = [table_features[i] for i in test_indexes]
    test_protein_table_features = [protein_table_features[i] for i in test_indexes]

    # if True:
    #     df_test_seqs=pd.DataFrame(test_seqs)
    #     df_test_seqs.to_csv("HCT-116_test_seqs.csv",index=False)


    if us_GNN:
        bpseq_folder="bpseq"
        map_path="data/mapinfo.pkl"
        if os.path.exists(map_path):
            with open(map_path,'rb') as f:
                gnndataset=pickle.load(f)
        else:
            gnndataset = build_dataset(bpseq_folder, label_data)
            with open(map_path,'wb') as f:
                pickle.dump(gnndataset,f)
        train_gnn_features = [gnndataset[i] for i in train_indexes]
        test_gnn_features = [gnndataset[i] for i in test_indexes]
    else:
        train_gnn_features=None
        test_gnn_features=None
    if us_DNASHAPE:
        shapeFeature=[read_dnashape_feature(dnashape_path+x,seq=" ",max_len=max_len) for x  in os.listdir(dnashape_path)]
        # with open("data/S2_shape.pkl",'rb') as f:
        #     shapeFeature=pickle.load(f)
        shapeFeature=[pd.DataFrame(shapeFeature[i]) for i in range(len(shapeFeature))]
        # dim reserve
        shapeFeature=np.dstack([df.values for df in shapeFeature])
        # Maximum length of interception
        # shapeFeature=shapeFeature[:,:max_len,:]
        shapeFeature=np.nan_to_num(shapeFeature,nan=0)
        # Training set test set division
        train_dnashape_feature=[shapeFeature[i] for i in train_indexes]
        test_dnashape_feature=[shapeFeature[i] for i in test_indexes]
    else:
        train_dnashape_feature=None
        test_dnashape_feature=None
    if us_Pretraining:
        # tokenizer=AutoTokenizer.from_pretrained(pretraining_path)
        # train_output=tokenizer(train_seqs,add_special_tokens=True,max_length=310,padding="longest",return_tensors='pt',truncation=True)
        # test_output=tokenizer(test_seqs,add_special_tokens=True,max_length=310,padding="longest",return_tensors='pt',truncation=True)
        train_tokens_ids,train_attention_mask=big_model.get_model_input(train_seqs)
        test_tokens_ids,test_attention_mask=big_model.get_model_input(test_seqs)
    else:
        train_output=None
        test_output=None
        train_tokens_ids=None
        test_tokens_ids=None
        train_attention_mask=None
        test_attention_mask=None

    train_dataset = CelllineDataset(train_indexes, train_seqs, train_protein_seqs, train_labels, train_fitness, train_table_features,train_protein_table_features,ftrain_seqs,emb_type,
                                    seq_type, max_len,protein_max_len,gnn_features=train_gnn_features,dnashape=train_dnashape_feature,tokens_ids=train_tokens_ids,attention_mask=train_attention_mask)
    test_dataset = CelllineDataset(test_indexes, test_seqs, test_protein_seqs, test_labels, test_fitness, test_table_features,test_protein_table_features,ftest_seqs,emb_type,
                                   seq_type, max_len,protein_max_len,gnn_features=test_gnn_features,dnashape=test_dnashape_feature,tokens_ids=test_tokens_ids,attention_mask=test_attention_mask)
    return train_dataset, test_dataset, col_len


# load_dataset("nucleotide", "onehot", 121, 42)


def load_test_dataset(seq_type, emb_type, max_len, seq_path, features_path):
    """ Load train & test dataset """
    seq_data, _, ccds_ids = load_data(seq_path)
    table_features = pd.read_csv(features_path, header=0)
    table_features = table_features.values.tolist()

    # split data with balanced test set
    test_indexes = range(len(seq_data))
    label_data = range(len(seq_data))
    test_seqs = [seq_data[i] for i in test_indexes]
    test_labels = np.array([label_data[i] for i in test_indexes])
    test_tabel_features = [table_features[i] for i in test_indexes]
    ftest_seqs=[Reverse(i) for i in test_seqs]


    test_dataset = CelllineDataset(test_indexes, test_seqs, test_labels, test_tabel_features,ftest_seqs,emb_type,
                                   seq_type, max_len)

    return test_dataset

def load_motif_dataset(seq_type, emb_type, max_len,feature_path, seed,dataset_path,DSEG_label=None,protein_feature_path=None):
    """ Load train & test dataset """
    protein_seq_data, seq_data, label_data = load_data_csv(dataset_path)
    print(len(protein_seq_data),len(seq_data),len(label_data))
    if DSEG_label!=None:
        label_data=DSEG_label
    table_features = pd.read_csv(feature_path, header=0)
    if protein_feature_path!=None:
        protein_table_features = pd.read_csv(protein_feature_path, header=0)
    else:
        protein_table_features = pd.DataFrame(index=range(len(label_data)))
    if 'label' in list(table_features.columns):
        table_features = table_features.drop(columns=["label"])
    col_len = len(table_features.columns)
    table_features = table_features.values.tolist()
    protein_table_features = protein_table_features.values.tolist()

    ess_indexes = [i for i, e in enumerate(label_data) if int(e) == 1]
    non_indexes = [i for i, e in enumerate(label_data) if int(e) == 0]
    num_ess = len(ess_indexes)
    # num_non = len(non_indexes)

    # split data with balanced test set
    r = 0.2
    random.seed(seed)
    random.shuffle(ess_indexes)
    random.shuffle(non_indexes)
    test_indexes = ess_indexes[:int(num_ess * r)] + non_indexes[:int(num_ess * r)]
    train_indexes = list(set(ess_indexes + non_indexes) - set(test_indexes))

    # with open("train_indexes.pkl","wb") as f:
    #     pickle.dump(train_indexes,f)
    # with open("test_indexes.pkl","wb") as f:
    #     pickle.dump(test_indexes,f)

    train_seqs = [seq_data[i] for i in train_indexes]
    train_protein_seqs = [protein_seq_data[i] for i in train_indexes]
    ftrain_seqs = [Reverse(i) for i in train_seqs]
    train_labels = np.array([label_data[i] for i in train_indexes])
    train_table_features = [table_features[i] for i in train_indexes]
    train_protein_table_features = [protein_table_features[i] for i in train_indexes]
    # print(len(train_table_features))

    test_seqs = [seq_data[i] for i in test_indexes]
    test_protein_seqs = [protein_seq_data[i] for i in test_indexes]
    ftest_seqs=[Reverse(i) for i in test_seqs]
    test_labels = np.array([label_data[i] for i in test_indexes])
    test_table_features = [table_features[i] for i in test_indexes]
    test_protein_table_features = [protein_table_features[i] for i in test_indexes]

    train_indexes.extend(test_indexes)
    train_seqs.extend(test_seqs)
    train_labels=np.append(train_labels,test_labels)
    train_table_features.extend(test_table_features)
    train_protein_table_features.extend(test_protein_table_features)
    ftrain_seqs.extend(ftest_seqs)
    train_protein_seqs.extend(test_protein_seqs)

    fitness=np.array([0 for _ in range(len(train_labels))])

    train_dataset = CelllineDataset(train_indexes, train_seqs, train_protein_seqs, train_labels,fitness, train_table_features,train_protein_table_features,ftrain_seqs,emb_type,seq_type, max_len,600)

    return train_dataset, col_len