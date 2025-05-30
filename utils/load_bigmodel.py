import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
import math
from torch.nn.functional import relu
from math import sqrt
from transformers import AutoModel,AutoTokenizer,AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,AutoConfig,AutoModelForCausalLM
from transformers.models.bert.configuration_bert import BertConfig
from multimolecule import RnaTokenizer, SpliceBertForContactPrediction
import os
# from evo import Evo
from tqdm import tqdm
from peft import LoraConfig, TaskType
from peft import get_peft_model

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


class BigPreTrainModel:
    def __init__(self,pretrain_path,pretrain_name,device):
        self.path=pretrain_path
        self.name=pretrain_name
        self.device=device
    def load_model(self):
        # 根据不同的模型加载不同的模型
        if self.name in ['hyenadna-tiny-16k-seqlen-d128-hf','hyenadna-tiny-1k-seqlen-d256-hf','hyenadna-tiny-1k-seqlen-hf','hyenadna-medium-160k-seqlen-hf',
                         'hyenadna-medium-450k-seqlen-hf','DNABERT-S','gena-lm-bert-base-t2t','hyenadna-small-32k-seqlen-hf']:
            self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True)
        elif self.name in ['nucleotide-transformer-500m-human-ref','nucleotide-transformer-v2-50m-multi-species','nucleotide-transformer-2.5b-multi-species',
                           'nucleotide-transformer-v2-250m-multi-species','nucleotide-transformer-2.5b-1000g','nucleotide-transformer-500m-1000g','gpn-msa-sapiens',
                           'nucleotide-transformer-v2-500m-multi-species','agro-nucleotide-transformer-1b','nucleotide-transformer-v2-100m-multi-species',
                           'PlantCaduceus_l20','gpn-brassicales'
                           ]:
            self.model = AutoModelForMaskedLM.from_pretrained(self.path, trust_remote_code=True)
        elif self.name in ['caduceus-ph_seqlen-131k_d_model-256_n_layer-16','caduceus-ps_seqlen-131k_d_model-256_n_layer-16','GROVER']:
            self.model = AutoModelForMaskedLM.from_pretrained(self.path, trust_remote_code=True)
        elif self.name in ['DNA_bert_3','DNA_bert_4','DNABERT-2-117M','DNA_bert_6','DNA_bert_5','DNABERT2-ME']:
            config = BertConfig.from_pretrained(self.path)
            self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True, config=config)
        elif self.name in ['splicebert.510nt','splicebert-human.510nt','splicebert']:
            self.model = SpliceBertForContactPrediction.from_pretrained(self.path,torch_dtype="auto")
        elif self.name in ['t5_baseline']:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path)
        elif self.name in ['evo-1-131k-base']:
            model_config = AutoConfig.from_pretrained(self.path, trust_remote_code=True, revision="1.1_fix")
            model_config.use_cache = True
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path,
                config=model_config,
                trust_remote_code=True,
                revision="1.1_fix"
            )
        self.model = self.model.to(self.device)
        if ("hyenadna" in self.name) or ("2.5b" in self.name):
            # 参数冻结，不参与训练
            for name,param in self.model.named_parameters():
                if 'proj' not in name:
                    param.requires_grad = False
            return self.model
            # print(self.model)
        print(self.model)
        # target_modules = [name for name, _ in self.model.named_modules() if ('proj' in name) or ('query' in name) or ('key' in name) or ('value' in name)]
        target_modules = [name for name, _ in self.model.named_modules() if ('proj' in name) or ('query' in name) or ('key' in name) or ('value' in name) or ('Wqkv' in name)] # DNABERT2
        # if len(target_modules)>10:
        #     target_modules=target_modules[:8]
        print("微调：",target_modules)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=target_modules)
        self.model = get_peft_model(self.model, peft_config)
        # model.print_trainable_parameters()
        return self.model
    def load_tokenizer(self):
        if self.name in ['hyenadna-tiny-16k-seqlen-d128-hf',"nucleotide-transformer-500m-human-ref",'DNA_bert_3','nucleotide-transformer-v2-50m-multi-species',
                         'nucleotide-transformer-2.5b-multi-species','nucleotide-transformer-v2-250m-multi-species','DNA_bert_4','nucleotide-transformer-2.5b-1000g',
                         'hyenadna-tiny-1k-seqlen-d256-hf','nucleotide-transformer-500m-1000g','gpn-msa-sapiens','nucleotide-transformer-v2-500m-multi-species',
                         'DNABERT-2-117M','hyenadna-tiny-1k-seqlen-hf','hyenadna-medium-160k-seqlen-hf','agro-nucleotide-transformer-1b','GROVER',
                         'hyenadna-medium-450k-seqlen-hf','DNA_bert_6','DNA_bert_5','nucleotide-transformer-v2-100m-multi-species','DNABERT-S',
                         'gena-lm-bert-base-t2t','caduceus-ph_seqlen-131k_d_model-256_n_layer-16','hyenadna-small-32k-seqlen-hf','caduceus-ps_seqlen-131k_d_model-256_n_layer-16',
                         'gpn-brassicales',"DNABERT2-ME"
                         ]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        elif self.name in ['splicebert.510nt','splicebert-human.510nt','splicebert']:
            self.tokenizer = RnaTokenizer.from_pretrained(self.path, trust_remote_code=True,torch_dtype="auto")
        elif self.name in ['t5_baseline']:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path+"/t5_regression_vocab", trust_remote_code=True)
        elif self.name in ['evo-1-131k-base']:
            self.tokenizer=AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        return self.tokenizer
    
    def get_model_input(self,input_dna):

        tokens_ids,attention_mask = None , None
        if self.name in ['hyenadna-tiny-16k-seqlen-d128-hf','hyenadna-tiny-1k-seqlen-d256-hf','hyenadna-tiny-1k-seqlen-hf','hyenadna-medium-160k-seqlen-hf',
                         'hyenadna-medium-450k-seqlen-hf','hyenadna-small-32k-seqlen-hf']:
            test_output=self.tokenizer(input_dna,add_special_tokens=True,max_length=512,padding="longest",return_tensors='pt',truncation=True)
            tokens_ids=test_output['input_ids']
        elif self.name in ['nucleotide-transformer-500m-human-ref','nucleotide-transformer-v2-50m-multi-species','nucleotide-transformer-2.5b-multi-species',
                           'nucleotide-transformer-v2-250m-multi-species','nucleotide-transformer-2.5b-1000g','nucleotide-transformer-500m-1000g',
                           'nucleotide-transformer-v2-500m-multi-species','nucleotide-transformer-v2-100m-multi-species']:
            tokens_ids = self.tokenizer.batch_encode_plus(input_dna, return_tensors="pt", padding="max_length", max_length = 512, truncation=True)["input_ids"]
            attention_mask = tokens_ids != self.tokenizer.pad_token_id
        elif self.name in ['DNA_bert_3','DNA_bert_4','DNA_bert_6','DNA_bert_5','t5_baseline']:
            if self.name=="DNA_bert_3":
                input_dna = [seq2kmer(seq,3) for seq in input_dna]
            tokens_ids = self.tokenizer.batch_encode_plus(input_dna, return_tensors="pt", padding="max_length", max_length = 512, truncation=True)["input_ids"]
            attention_mask = tokens_ids != self.tokenizer.pad_token_id
        elif self.name in ['DNABERT-2-117M','DNABERT-S','gena-lm-bert-base-t2t','DNABERT2-ME']:
            tokens_ids = self.tokenizer.batch_encode_plus(input_dna, return_tensors="pt", padding="max_length", max_length = 512, truncation=True)["input_ids"]
            attention_mask = tokens_ids != self.tokenizer.pad_token_id
        elif self.name in ['splicebert.510nt','splicebert-human.510nt','splicebert']:
            tokens_ids = self.tokenizer.batch_encode_plus(input_dna, return_tensors="pt", padding="max_length", max_length = 512, truncation=True)["input_ids"]
            attention_mask = tokens_ids != self.tokenizer.pad_token_id
        elif self.name in ['agro-nucleotide-transformer-1b']:
            tokens_ids = self.tokenizer.batch_encode_plus(input_dna, return_tensors="pt", max_length = 512, truncation=True, padding=True)["input_ids"]
            attention_mask = tokens_ids != self.tokenizer.pad_token_id
        elif self.name in ['caduceus-ph_seqlen-131k_d_model-256_n_layer-16','caduceus-ps_seqlen-131k_d_model-256_n_layer-16','GROVER']:
            tokens_ids = self.tokenizer.batch_encode_plus(input_dna, return_tensors="pt", max_length = 512, truncation=True, padding=True)["input_ids"]
        elif self.name in ['gpn-brassicales']:
            tokens_ids = self.tokenizer(input_dna, return_tensors="pt", max_length = 512,return_attention_mask=False, return_token_type_ids=False, truncation=True, padding=True)["input_ids"]
            attention_mask = tokens_ids != self.tokenizer.pad_token_id
        return tokens_ids,attention_mask
    def get_model_output(self,tokens_ids,attention_mask):
        if self.name in ['hyenadna-tiny-16k-seqlen-d128-hf','hyenadna-tiny-1k-seqlen-d256-hf','hyenadna-tiny-1k-seqlen-hf','hyenadna-medium-160k-seqlen-hf',
                         'hyenadna-medium-450k-seqlen-hf','hyenadna-small-32k-seqlen-hf']:
            output=self.model(tokens_ids).last_hidden_state
        elif self.name in ['nucleotide-transformer-500m-human-ref','nucleotide-transformer-v2-50m-multi-species','nucleotide-transformer-2.5b-multi-species',
                           'nucleotide-transformer-v2-250m-multi-species','nucleotide-transformer-2.5b-1000g','nucleotide-transformer-500m-1000g',
                           'nucleotide-transformer-v2-500m-multi-species','nucleotide-transformer-v2-100m-multi-species']:
            output=self.model(tokens_ids,attention_mask=attention_mask,encoder_attention_mask=attention_mask,output_hidden_states=True)['hidden_states'][-1]
        elif self.name in ['DNA_bert_3','DNA_bert_4','DNA_bert_6','DNA_bert_5','t5_baseline']:
            output=self.model(tokens_ids,attention_mask=attention_mask,encoder_attention_mask=attention_mask,output_hidden_states=True).last_hidden_state
        elif self.name in ['DNABERT-2-117M','DNABERT-S','gena-lm-bert-base-t2t','DNABERT2-ME']:
            output=self.model(tokens_ids,attention_mask=attention_mask,encoder_attention_mask=attention_mask,output_hidden_states=True)[0]
        elif self.name in ['splicebert.510nt','splicebert-human.510nt','splicebert']:
            output=self.model(tokens_ids,attention_mask=attention_mask,encoder_attention_mask=attention_mask,output_hidden_states=True).logits
        elif self.name in ['agro-nucleotide-transformer-1b']:
            output=self.model(tokens_ids,attention_mask=attention_mask,output_hidden_states=True).logits
            # print(output)
        elif self.name in ['caduceus-ph_seqlen-131k_d_model-256_n_layer-16','caduceus-ps_seqlen-131k_d_model-256_n_layer-16','GROVER']:
            output=self.model(tokens_ids,output_hidden_states=True).logits
        elif self.name in ['gpn-brassicales']:
            output=self.model(tokens_ids)
        return output