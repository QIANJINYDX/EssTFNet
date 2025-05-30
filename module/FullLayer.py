import torch
import torch.nn as nn
import torch.nn.functional as F
class FullLayer(nn.Module):
    def __init__(self, input_dim):
        super(FullLayer,self).__init__()
        # 定义残差块
        self.residual_block = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim),
            nn.ReLU()
        )
        # 定义全连接层和Dropout层
        self.generator = nn.Sequential(nn.Linear(input_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, 1),
                                       nn.Dropout(0.2),
                                       nn.Sigmoid())
        self.dropout = nn.Dropout(0.2)
        # 定义Sigmoid层
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        residual_output = self.residual_block(x)
        x = x + residual_output
        x = self.generator(x)
        return x
