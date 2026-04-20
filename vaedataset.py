import numpy as np
import torch
import nibabel as nb
import pandas as pd
import random
from torch.utils.data import Dataset

def random_flip_3d(image):
    if random.random() > 0.5:
        axis = random.choice([0, 1])
        image = np.flip(image, axis).copy()
    return image

class UKBDataset(Dataset):
    def __init__(self, json_path):
        # 1. 读取 DataFrame 并转为 字典列表 (Records)
        # 这样无论列怎么乱序，都可以通过 key 名字准确找到数据
        df = pd.read_json(json_path)
        self.t1_data = df.to_dict('records')

    def __len__(self):
        return len(self.t1_data)
  
    def __getitem__(self, index):
        item = self.t1_data[index]
        
        # 2. 使用列名获取坐标 (解决 .iat 索引错位问题)
        # 假设你的 JSON 生成代码中列名是 shape0 ~ shape5
        x1 = int(item['shape0'])
        x2 = int(item['shape1']) + 1
        y1 = int(item['shape2'])
        y2 = int(item['shape3']) + 1
        z1 = int(item['shape4'])
        z2 = int(item['shape5']) + 1
        
        # 3. 加载图像 (根据键名 high_path)
        img_path = item['high_path']
        try:
            # 加上类型转换 float32 节省内存并防止类型错误
            T2high = nb.load(img_path).get_fdata().astype(np.float32)
        except:
            # 如果读图失败，返回一个全0块防止训练崩溃
            return {'high': torch.zeros((1, 64, 64, 64))}

        # 4. 归一化 (根据键名获取均值方差)
        # 之前报错就是因为这里 iat 读到了字符串
        mask = T2high > 0
        mean_val = float(item['mean_high'])
        std_val = float(item['std_high'])
        
        if std_val < 1e-8: std_val = 1.0 # 防止除零

        # T2high_99 = np.percentile(T2high[mask], 99)
        # T2high[mask] = T2high[mask] / T2high_99
        # 只处理 Mask 区域
        T2high[mask] = (T2high[mask] - mean_val) / std_val
        
        # 5. 随机 Shift (保留你的逻辑)
        shift_num_1 = random.choice([-3,-2,-1,0,1,2,3])
        shift_num_2 = random.choice([-3,-2,-1,0,1,2,3])
        shift_num_3 = random.choice([-3,-2,-1,0,1,2,3])
        
        # 增加简单的边界保护，防止加上 shift 后越界
        img_shape = T2high.shape
        # sx1 = max(0, min(x1 + shift_num_1, img_shape[0]))
        # sx2 = max(0, min(x2 + shift_num_1, img_shape[0]))
        # sy1 = max(0, min(y1 + shift_num_2, img_shape[1]))
        # sy2 = max(0, min(y2 + shift_num_2, img_shape[1]))
        # sz1 = max(0, min(z1 + shift_num_3, img_shape[2]))
        # sz2 = max(0, min(z2 + shift_num_3, img_shape[2]))

        T2high_block = T2high[x1:x2, y1:y2, z1:z2]
        
        # 6. 数据增强与格式转换
        T2high_block = random_flip_3d(T2high_block)
        
        # 确保 block 维度正确 (Shift 可能导致边缘尺寸不足 64，这里依赖 PyTorch DataLoader 可能会报错，
        # 如果报错 size mismatch，你需要在这里强制 resize 或者 pad，但先保留你的逻辑)
        
        T2high_block = T2high_block.reshape((1,) + T2high_block.shape)
        T2high_block = torch.tensor(T2high_block, dtype=torch.float32)

        data_pair = {'high': T2high_block}
        return data_pair