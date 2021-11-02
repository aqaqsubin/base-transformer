
import pandas as pd
import torch

def to_cuda(batch, gpuid):
    device = torch.device('cuda:%d' % gpuid)
    for i, n in enumerate(batch):
        if n != "data":
            batch[n] = batch[n].to(dtype=torch.long, device=device)

def saveCSVFile(filepath, dist):
    dist.to_csv(filepath, mode='w', sep='\t', index=False, encoding='utf-8')

def saveXLSXFile(filepath, dist):
    with pd.ExcelWriter(filepath) as writer:
        dist.to_excel(writer, sheet_name='sheet_name_1',engine='xlsxwriter')

def get_filetype(filepath):
    return filepath.split('/')[-1].split('.')[1]

def read_df(filepath, sep='\t'):
    filetype = get_filetype(filepath)
    if filetype == 'csv':
        data = pd.read_csv(filepath, sep=sep)
    elif filetype == 'xlsx':
        data = pd.read_excel(filepath, engine='openpyxl')
    return data

def collate_mp(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

