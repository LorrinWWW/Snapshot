import torch
import torch.nn as nn
import torch.functional as F
import transformers
import numpy as np


class bert(nn.Module):

    def __init__(self, ckpt_path):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(ckpt_path)

    def forward(self, x):
        _, pooled = self.bert(**x, return_dict = False)
        return pooled