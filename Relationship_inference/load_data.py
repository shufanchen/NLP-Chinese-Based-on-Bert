#coding=utf-8
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random


class TextDataset(Dataset):
    def __init__(self, dataframe):
        # 只保留长度大于40的文本
        self.dataframe = dataframe[dataframe.iloc[:, 0].str.len() > 40]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx, 0]  # 第一列是text
        sentence1 = text[:20]
        sentence2 = text[20:]
        label = 0
        if random.randint(0, 1) == 0:
            j = random.randint(0, len(self.dataframe) - 1)
            sentence2 = self.dataframe.iloc[j,0][20:40]
            label = 1
        text = sentence1+sentence2
        sample = {"text": text, "label": label}
        return sample



def make_loader(path='../data/ChnSentiCorp.csv'):
    # 第一步：读取CSV文件
    df = pd.read_csv(path).iloc[:,1:]
    # 第二步：定义自定义Dataset类
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)  # 80%训练，20%临时测试
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    # 第三步：创建Dataset实例
    # 创建训练集、验证集和测试集的 Dataset
    train_dataset = TextDataset(train_df)
    val_dataset = TextDataset(val_df)
    test_dataset = TextDataset(test_df)
    # 创建 DataLoader
    # 你可以根据需要调整 batch_size 和是否打乱数据（shuffle）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader,val_loader,test_loader





