# **EssTFNet: Temporal-Frequency Deep Learning Integrated with DNA Large Language Models for Human Essential Gene Prediction** 

<img width="4803" height="2883" alt="人类重要基因预测-第 18 页" src="https://github.com/user-attachments/assets/8947e2c0-fa84-438b-8e2a-aa5b5c7c11d3" />


A novel deep learning framework that integrates **temporal-frequency signal processing** with **DNA-specific large language models** to predict human essential genes from genomic sequences.

## 🌟 Key Features

- **ATFNet**: Optimization of Long Sequence Modeling Problems Using ATFNet Networks
- **DNA-Specific Embeddings**: Utilizes pre-trained DNA-LLM (e.g., Nucleotide Transformer) for biological sequence representation
- **Interpretability**: Base Sequence Extraction of Different Lengths Using DeepLift Combined with Low-Pass Filtering

## 🚀 Quick Start

**Installation**

```
git clone https://github.com/yourusername/EssTFNet.git
cd EssTFNet
conda create -n EssTFNet python=3.11
conda activate EssTFNet
pip install -r requirements.txt
```

**Data Preparation**

Perform feature extraction by running `featureExtraction.ipynb`

Obtain the feature file.

**Training**

Run the script

```
python main_single.py \
    --max_len 1800 \
    --protein_max_len 600 \
    --dataset_path "data/S1.csv" \
    --feature_path "data/S1_feature/feature4800.csv" \
    --batch_size 16 \
    --is_double_stranded False \
    --is_only_feature False \
    --us_GNN False \
    --us_DNASHAPE False \
    --us_Pretraining False \
    --patience 2 \
    --gpu 0 \
    --result_path "result/S1_classification.csv" \
    --seed 114514 \
    --is_balance_data False \
    --is_ATFnet True \
    --is_motif_extraction False \
    --task_type "classification" 
```

Fine-tune the pre-trained model

```
python main_single.py \
    --max_len 1800 \
    --protein_max_len 600 \
    --dataset_path "data/S4.csv" \
    --feature_path "data/S4_feature/final/feature4800.csv" \
    --batch_size 8 \
    --is_double_stranded False \
    --is_only_feature False \
    --us_GNN False \
    --us_DNASHAPE False \
    --us_Pretraining True \
    --pretraining_path "pretraining_path" \
    --pretraining_name "nucleotide-transformer-v2-100m-multi-species" \
    --patience 5 \
    --gpu 1 \
    --result_path "result/S4_pretraining.csv" \
    --seed 114514 \
    --is_balance_data False \
    --is_ATFnet True \
    --is_motif_extraction False \
    --task_type "classification" \
    --save_path "saved_model/S4/pretraining" \
    --us_protein True
```



