import glob
import os
import re
from pathlib import Path
import torch
import numpy as np

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    dir = path if path.suffix == "" else path.parent
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)
    return path
class Normalizer:
    def __init__(self, data):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
            
        self.mean = np.mean(data, axis=(0, 1)).astype(np.float32)
        self.std = np.std(data, axis=(0, 1)).astype(np.float32)
        self.std[self.std == 0] = 1.0

    def normalize(self, x):
        if torch.is_tensor(x):
            # 恢复全局归一化：让包含 6D 旋转在内的所有 151 维数据处于同一量纲
            mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.tensor(self.std, device=x.device, dtype=x.dtype)
            return (x - mean) / std
        else:
            return (x - self.mean) / self.std

    def unnormalize(self, x):
        if torch.is_tensor(x):
            mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.tensor(self.std, device=x.device, dtype=x.dtype)
            return x * std + mean
        else:
            return x * self.std + self.mean
        
def vectorize_many(data_list):
    """
    具备维度自适应展平能力的特征拼接器。
    支持将高维旋转张量(如[B, S, Joint, 6])安全展平并拼接。
    """
    processed_list = []
    for d in data_list:
        if not torch.is_tensor(d):
            d = torch.tensor(d)
        
        # 核心逻辑：如果维度大于 3 (例如 4维的旋转张量 [B, S, Joint, 6])
        # 强制将其后两维展平为特征维 [B, S, Joint*6]
        if len(d.shape) > 3:
            b, s = d.shape[:2]
            d = d.reshape(b, s, -1)
        # 如果是 [B, S] 这种漏掉特征维的，补上一维
        elif len(d.shape) == 2:
            d = d.unsqueeze(-1)
            
        processed_list.append(d)
    
    # 此时所有张量都是 3 维 (Batch, Seq, Feats)，可以安全拼接
    return torch.cat(processed_list, dim=-1)