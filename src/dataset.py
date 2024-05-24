import os
import json
import torch
from torch.utils.data import Dataset
from src.utils import table2string
from itertools import chain

class ClinicalDataset(Dataset):
    def __init__(self,header,table_value,table_label,eos_token,baseline):
        self.table_value = table_value
        self.table_label = table_label[0]
        self.action_label = table_label[1]
        self.table_strings = table2string(header,self.table_value,eos_token,baseline)
        self.baseline = baseline
    def __getitem__(self,index):
        return self.table_strings[index],self.table_label[index],self.action_label[index]

    
    def __len__(self):
        return len(self.table_strings)

    def batchify(self, batch, tokenizer):
        tokens, label,action_label = map(list, zip(*batch))
        text_ids = tokenizer(tokens,
                            padding=True,
                            truncation=True,
                            max_length=656,
                            add_special_tokens=False,
                            return_tensors='pt')
        if self.baseline:
            return text_ids, torch.tensor(label)            
        else:
            return text_ids,torch.tensor(list(chain.from_iterable(action_label))),torch.tensor(label)

