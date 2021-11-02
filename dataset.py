
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

class MathData(Dataset):
    def __init__(self, filepath, target, max_len, **kwargs):
        self._data = pd.read_csv(filepath, sep='\t')
        self.first = True
        self.max_len = max_len
        self.target = target

        self.tokenizer = kwargs['tokenizer'] 
        self.bos = self.tokenizer.eos_token
        self.eos = self.tokenizer.eos_token
        self.pad = self.tokenizer.pad_token
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['question']
        a = turn[self.target]

        q_toked = self.tokenizer.tokenize(self.bos + str(q) + self.eos)
        q_len = len(q_toked)
        if q_len > self.max_len:
            q_toked = q_toked[:self.max_len-1] + q_toked[-1]
            q_len = len(q_toked)
            assert q_len == len(q_toked), f'{q_len} ==? {len(q_toked)}'
        
        a_toked = self.tokenizer.tokenize(self.bos + str(a) + self.eos)
        a_len = len(a_toked)
        if a_len > self.max_len:
            a_toked = a_toked[:self.max_len-1] + a_toked[-1]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'

        labels_ids = self.tokenizer.convert_tokens_to_ids(a_toked)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, labels_ids)